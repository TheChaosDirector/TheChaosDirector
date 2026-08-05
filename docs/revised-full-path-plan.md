# Revised plan — full path (live Gmail + Drive)

You now have **Gmail** and **Google Drive** connected. We upgrade from the A234 “everything mocked” prototype to the **real automation spine**, with only Mercury still faked until you have Bill Pay access.

## What’s live vs mocked now

| Step | Status |
|---|---|
| Invoice hits finance Gmail | **LIVE** — Gmail trigger |
| Save PDF to Drive | **LIVE** |
| Master Tracker Google Sheet | **LIVE** |
| Vendor / W-9 check (Sheet lookup) | **LIVE** (directory tab) |
| Auto-reply asking for W-9 | **LIVE** Gmail reply (or draft) |
| AI extract invoice fields | **LIVE** (Zapier AI / ChatGPT action) |
| Stacey sets `3 - Ready` in Sheet | **LIVE** (human) |
| Head approval email w/ Approve + Reject | **LIVE** Gmail |
| Approve/Reject webhook → Sheet update | **LIVE** |
| Create Mercury Bill Pay draft | **MOCK** → Mercury Log tab (+ optional email-to-self “forwarded to Mercury”) |
| Stacey Pay in Mercury | **MOCK** → checkbox / Pay link until Mercury exists |
| Settled → `5 - Paid` + archive | **LIVE** Sheet update (triggered by mock Pay for now) |
| Friday digest + monthly archive review | **LIVE** Gmail |

When Mercury is available later: replace the Mercury Log step with “email PDF to Mercury Bill Pay inbox” and optionally Mercury Settled Transaction → Paid.

---

## Apps / accounts needed

Already connected (per you):
- Zapier
- Gmail
- Google Drive

Still need in Zapier:
- **Google Sheets** (same Sheet as Master Tracker)
- **Webhooks by Zapier** (Approve/Reject Catch Hook)
- **ChatGPT / Zapier AI** (or PDF parser) for extraction
- Optional: Typeform for W-9 uploads

Still mocked:
- Mercury Bill Pay

---

## Sheet tabs (source of truth)

1. **Master Tracker** — active invoices  
2. **Vendor Directory** — W-9 on file?  
3. **Archive** — paid / closed  
4. **Mercury Log** — mock drafts + mock payments (temporary)  
5. **Email Log** — optional audit copy of outbound mail  

Use columns from `docs/templates/master-tracker-headers.csv` (+ `Pay Mock` checkbox column for prototype Pay).

---

## Zap map (full path)

### Zap 1 — Intake (Gmail → Drive → Sheet)
- **Trigger:** Gmail — New email matching search  
  Example: `to:finance@... has:attachment filename:pdf`
- **Action:** Drive — Upload file (PDF attachment) to `Invoices/Incoming`
- **Action:** Sheets — Lookup vendor in Vendor Directory
- **Paths:**
  - **Missing W-9:** Gmail reply with Typeform/W-9 link → create Sheet row Status `2 - Waiting on Docs`
  - **W-9 OK:** continue to Zap 2 / same Zap Path → AI extract → create row Status `1 - Intake` (or ready for Stacey)

### Zap 2 — Tax form received (optional Typeform)
- **Trigger:** Typeform submission  
- Update Vendor Directory `W-9 on File = Yes`  
- Find waiting rows for vendor → set Intake + notify Stacey

### Zap 3 — AI parse (if not folded into Zap 1)
- Extract Vendor, Invoice #, Amount, Due Date  
- Update Sheet row + Drive link

### Zap 4 — Ready → Head approval email
- **Trigger:** Sheets — Updated row where Status = `3 - Ready`
- **Action:** Gmail — Send email to Head of School  
  Body: vendor/amount/PDF link + HTML **Approve** / **Reject** buttons  
  Links hit Catch Hook with `decision` + `row_id` (+ secret token)

### Zap 5 — Decision webhook
- **Trigger:** Webhooks — Catch Hook
- **Approve path:**
  - Update row → `4 - Approved` + timestamp  
  - Append **Mercury Log** `DRAFT_CREATED` (mock)  
  - Optional: Gmail to Stacey “Approved — mock Mercury draft ready”
- **Reject path:**
  - Update row → `6 - Rejected` + reason  
  - Gmail Stacey — do **not** touch Mercury Log as draft

### Zap 6 — On Hold
- **Trigger:** Schedule daily  
- Find `3 - Ready` older than 3 business days → `7 - On Hold`  
- Gmail reminder to Head (CC Stacey)

### Zap 7 — Mock Pay → Paid + archive
Until Mercury exists, pick one:
- **A.** Sheets — checkbox column `Pay Mock` checked on Approved row, or  
- **B.** Second webhook “PAY” link emailed to Stacey after Approve  
- Then: Status `5 - Paid` + Date Paid → Mercury Log `PAYMENT_SETTLED` → move row to Archive

### Zap 8 — Friday digest
- Schedule Friday → Gmail summary of Paid / Ready / On Hold / Rejected

### Zap 9 — Monthly archive review
- Schedule 1st (or 1st+15th) → Gmail Stacey checklist + Archive tab link

---

## Demo script (what you show Stacey)

1. Send a test PDF invoice **to the finance Gmail**  
2. Watch Drive folder + Master Tracker row appear  
3. (If new vendor) see W-9 auto-reply  
4. Stacey sets Status `3 - Ready`  
5. Head gets **real Gmail** with Approve/Reject  
6. Click Approve → Sheet `4` + Mercury Log draft line  
7. Stacey “Pays” via mock control → `5 - Paid` + Archive  
8. Show Friday/monthly emails as scheduled

---

## What we are NOT doing anymore

- Apps Script menu clicking as the product  
- HTML-only fake walkthrough as the prototype  
- Mocking Gmail/Drive (those are live now)

---

## Implementation note (this cloud agent)

Even with Zapier connected in **Cursor desktop**, this **cloud agent session** cannot call Zapier MCP (`needsAuth` here / no interactive MCP auth).

**To actually create the Zaps with the agent:**
1. Open this repo/chat in **Cursor desktop** (where Zapier MCP shows connected/tools available)
2. Ask: “Build the full-path Zaps from `docs/revised-full-path-plan.md`”
3. Or build manually in zapier.com using the Zap map above

---

## Mercury upgrade later (5 minutes when you have it)

Replace Zap 5 Approve → Mercury Log with:
- Gmail/Email: send PDF to Mercury Bill Pay inbox address  

Replace Zap 7 mock Pay with:
- Mercury Settled Transaction → match row → Paid + archive
