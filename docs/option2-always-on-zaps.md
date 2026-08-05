# Option 2 — Always-on Zaps (zapier.com)

Build these on **[zapier.com](https://zapier.com)** so they run automatically.  
Zapier MCP chat skills are **not** this plan.

## Your live assets (already created)

| Asset | Link / ID |
|---|---|
| **Google Sheet** | https://docs.google.com/spreadsheets/d/148VDOsPeqWq3XCQw39WlFhp-vN7ZlblFsQS_PuxWcgQ/edit |
| Spreadsheet ID | `148VDOsPeqWq3XCQw39WlFhp-vN7ZlblFsQS_PuxWcgQ` |
| **Drive folder** | https://drive.google.com/drive/folders/109Fi3y9EXu85RCSwu-wEPNbsCuB73KPp |
| **Incoming PDFs** | https://drive.google.com/drive/folders/1wB7AT4kN5hXMxrXZhcvQukR89sQul_w- |
| Incoming folder ID | `1wB7AT4kN5hXMxrXZhcvQukR89sQul_w-` |
| Gmail account (Zapier) | `benjamimyam123@gmail.com` |

### Sheet tabs
| Tab | Purpose |
|---|---|
| Master Tracker | Active invoices |
| Vendor Directory | W-9 status (seeded with 2 demo vendors) |
| Archive | Paid / closed rows |
| Mercury Log | **Mock** Mercury drafts + payments until Bill Pay exists |

### Master Tracker columns
`Row ID` · `Date Received` · `Vendor` · `Invoice #` · `Amount` · `Due Date` · `Drive Link` · `Status` · `Tax / W-9 OK?` · `Approved At` · `Rejected At` · `Reject Reason` · `Date Paid` · `Pay Mock` · `Last Archive Review` · `Notes` · `Approve Token`

### Statuses
`1 - Intake` · `2 - Waiting on Docs` · `3 - Ready` · `4 - Approved` · `5 - Paid` · `6 - Rejected` · `7 - On Hold`

**Hard rule:** only `4 - Approved` writes a Mercury Log draft (mock). Rejected never does.

---

## Before you build Zaps

1. Open the Sheet link above — confirm tabs exist.  
2. In Zapier: connect **Gmail**, **Google Drive**, **Google Sheets**, **Webhooks by Zapier**.  
3. Optional: **ChatGPT / Zapier AI** (or PDF4me / Parseur) for invoice extraction.  
4. Decide emails:
   - **Finance inbox** = the Gmail Zap watches (likely `benjamimyam123@gmail.com` for testing)
   - **Head of School** = who gets Approve/Reject
   - **Stacey** = who gets reject notices / digests  
5. Optional: create a Typeform for W-9 (Zap 2). Can skip first week.

---

## Zap 1 — Intake (Gmail → Drive → Sheet)

### Trigger
- App: **Gmail**
- Event: **New Email Matching Search**
- Search: `has:attachment filename:pdf`  
  (tighten later, e.g. `subject:(invoice OR bill) has:attachment filename:pdf`)

### Actions
1. **Google Drive — Upload File**  
   - File: Gmail attachment  
   - Folder: `Incoming` (`1wB7AT4kN5hXMxrXZhcvQukR89sQul_w-`)
2. **Google Sheets — Lookup Spreadsheet Row**  
   - Spreadsheet: Invoice Approval Tracker  
   - Worksheet: Vendor Directory  
   - Lookup column: Vendor Name  
   - Lookup value: best-effort vendor from email From/Subject (or AI step first)
3. **Paths**
   - **Path A — W-9 missing / no vendor row**
     - Gmail: Reply or Send — W-9 / Typeform link  
     - Sheets: Create Spreadsheet Row on Master Tracker  
       - Status = `2 - Waiting on Docs`  
       - Tax / W-9 OK? = `No`  
       - Drive Link = uploaded file URL  
       - Row ID = e.g. `INV-{{zap_id}}-{{timestamp}}`
   - **Path B — W-9 OK**
     - AI extract: Vendor, Invoice #, Amount, Due Date  
     - Sheets: Create Spreadsheet Row  
       - Status = `1 - Intake`  
       - Tax / W-9 OK? = `Yes`  
       - map extracted fields + Drive Link

### Test
Send yourself a PDF email with subject `Invoice INV-TEST-001`. Confirm file in Incoming + row on Master Tracker.

---

## Zap 2 — Tax form received (optional)

### Trigger
Typeform / Google Form new submission

### Actions
1. Sheets — Update Vendor Directory (W-9 on File = Yes, date)  
2. Sheets — Find Master Tracker rows for that vendor with Status `2 - Waiting on Docs`  
3. Update those rows → `1 - Intake`, Tax = Yes  
4. Gmail Stacey: “Docs received for {{Vendor}}”

Skip until W-9 form exists; for demos, manually set Tax = Yes and Status = Intake.

---

## Zap 3 — Ready → Head approval email

### Trigger
- Google Sheets — **New or Updated Spreadsheet Row**
- Spreadsheet: Invoice Approval Tracker  
- Worksheet: Master Tracker  
- Trigger column: Status  
- Only continue if Status **exactly** `3 - Ready` (Filter)

### Actions
1. Formatter / Code: generate `Approve Token` if empty (UUID) — or set token when Stacey flips Ready  
2. **Webhooks by Zapier** — note: Catch Hook URL comes from Zap 4 (create Zap 4 first, copy URL here)
3. **Gmail — Send Email** to Head of School  
   - Subject: `Invoice approval needed — {{Vendor}} — {{Amount}}`  
   - Body type: HTML  
   - Include PDF Drive Link  
   - Buttons:

```html
<a href="CATCH_HOOK_URL?decision=approve&row_id={{Row ID}}&token={{Approve Token}}">APPROVE</a>
<a href="CATCH_HOOK_URL?decision=reject&row_id={{Row ID}}&token={{Approve Token}}">REJECT</a>
```

### How Stacey uses it
On Master Tracker, set Status to `3 - Ready` (and fill Approve Token if you want). Zap 3 sends the email.

---

## Zap 4 — Decision webhook (build this before finishing Zap 3 links)

### Trigger
- **Webhooks by Zapier — Catch Hook**
- Copy the Custom Webhook URL into Zap 3 email buttons

### Actions
1. Filter / Paths on query param `decision`
2. **Approve path**
   - Sheets: Lookup row by Row ID  
   - Sheets: Update row → Status `4 - Approved`, Approved At = now  
   - Sheets: Create row on **Mercury Log**  
     - Action = `DRAFT_CREATED`  
     - Status = `Draft (mock)`  
     - Notes = `Pretend PDF forwarded to Mercury Bill Pay inbox`
   - Gmail Stacey: “Approved — mock Mercury draft ready. Check Pay Mock when paid.”
3. **Reject path**
   - Update row → `6 - Rejected`, Rejected At, Reject Reason if present  
   - Gmail Stacey: rejected notice  
   - **Do not** write Mercury Log draft

### Test
Use Zapier’s “Test trigger” or open the Approve link from a real Zap 3 email.

---

## Zap 5 — On Hold

### Trigger
Schedule by Zapier — Every day

### Actions
1. Sheets — Get/Find rows where Status = `3 - Ready` and Date Received / Ready age > 3 business days  
   (Practical approach: add column `Ready At` when status becomes Ready; filter on that.)  
2. Looping / multiple updates → Status `7 - On Hold`  
3. Gmail Head (CC Stacey): reminder with same Approve/Reject links

---

## Zap 6 — Mock Pay → Paid + Archive

Until Mercury exists:

### Trigger option A (simplest)
- Sheets — New or Updated Row on Master Tracker  
- Filter: Status = `4 - Approved` AND `Pay Mock` = `Yes` (or `TRUE` / `x`)

### Actions
1. Update row → Status `5 - Paid`, Date Paid = today  
2. Mercury Log row: `PAYMENT_SETTLED`  
3. Create row on **Archive** with mapped fields + Archived At  
4. Delete/clear the Master Tracker row (or leave and filter “active only”)

### Stacey action
After Approve, open Sheet → set **Pay Mock** = Yes on that row.

---

## Zap 7 — Friday digest

### Trigger
Schedule — Every Friday 4pm (or your timezone)

### Actions
1. Pull counts from Master Tracker / Archive (Paid this week, Ready, On Hold, Rejected)  
2. Gmail Stacey (+ optional Head) with summary + Sheet link

---

## Zap 8 — Monthly archive review

### Trigger
Schedule — 1st of month (or 1st + 15th)

### Actions
Gmail Stacey: open Archive tab, spot-check paid/rejected, flag issues.

---

## Recommended build sequence

1. Create **Zap 4** Catch Hook first → copy URL  
2. Build **Zap 1** Intake → test with PDF email  
3. Manually set a row to `3 - Ready`  
4. Build **Zap 3** approval email using Catch Hook URL  
5. Click Approve/Reject in the email → confirm Sheet + Mercury Log  
6. Build **Zap 6** mock Pay  
7. Add On Hold + digests last  

Turn Zaps **ON** only after each test passes.

---

## Smoke test checklist

- [ ] PDF email → file in Incoming folder  
- [ ] Master Tracker row created  
- [ ] Missing-vendor path sends W-9 mail (or Path B works for known vendor)  
- [ ] Status `3 - Ready` → Head gets Gmail with two buttons  
- [ ] Approve → `4 - Approved` + Mercury Log `DRAFT_CREATED`  
- [ ] Reject → `6 - Rejected`, no Mercury draft  
- [ ] Pay Mock → `5 - Paid` + Archive row  
- [ ] Friday / monthly schedules exist (can leave OFF until ready)

---

## Mercury upgrade (later)

| Now | Later |
|---|---|
| Zap 4 writes Mercury Log `DRAFT_CREATED` | Email/forward PDF to Mercury Bill Pay inbox |
| Zap 6 Pay Mock checkbox | Mercury Settled Transaction → Paid + archive |

---

## What this agent cannot do

Zapier MCP **cannot create always-on Zaps** on zapier.com.  
This repo + your Sheet/Drive are ready; **you click Build Zap** on zapier.com using this guide.

If you get stuck on a specific Zap step, screenshot it and ask in Agent mode.
