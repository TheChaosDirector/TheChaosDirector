# Real Google Sheet prototype (email + Mercury mocked)

This runs the **full path on a real Google Sheet**.

| Piece | Real or mock? |
|---|---|
| Master Tracker statuses | **Real Sheet** |
| Approve / Reject links | **Real** (Apps Script web app) |
| Archive move | **Real Sheet** |
| Finance inbox / Head email | **Mock** → written to **Email Log** tab |
| Mercury Bill Pay | **Mock** → written to **Mercury Log** tab |

No Mercury account. No finance mailbox required.

---

## 10-minute setup

### 1) Create a Google Sheet
1. Go to [sheets.new](https://sheets.new)
2. Name it something like `Invoice Prototype`

### 2) Paste the script
1. In the Sheet: **Extensions → Apps Script**
2. Delete any stub code
3. Copy everything from [`sheet-prototype/Code.gs`](./sheet-prototype/Code.gs)
4. Paste into `Code.gs`
5. Click **Save** (disk icon)
6. Close the Apps Script tab and refresh the Sheet

### 3) Run setup from the Sheet menu
After refresh you should see **Invoice Prototype** in the menu bar.

1. **Invoice Prototype → 1) Setup sheet tabs**
2. **Invoice Prototype → 2) Seed sample vendors**
3. Authorize when Google asks (Review permissions → your account → Allow)

You should now have tabs:
- Master Tracker
- Vendor Directory
- Archive
- Email Log
- Mercury Log

### 4) Deploy Approve / Reject links (one time)
1. **Extensions → Apps Script**
2. **Deploy → New deployment**
3. Type: **Web app**
4. Description: `invoice prototype`
5. Execute as: **Me**
6. Who has access: **Anyone** (prototype only — fine for demo links)
7. **Deploy** → copy the URL if shown → **Done**
8. Back in the Sheet: **Invoice Prototype → Show Approve/Reject web app URL** (should show a URL)

---

## Demo the full path (happy path)

1. **Invoice Prototype → Inject TEST invoice (known vendor)**  
   → new row at `1 - Intake` + mock “inbox” line in Email Log
2. Click any cell on that Master Tracker row  
3. **Invoice Prototype → Stacey: mark selected row Ready**  
   → status `3 - Ready` + mock Head email in **Email Log** with Approve/Reject links
4. Open **Email Log** → find the latest approval email → click **APPROVE** link  
   → browser says Approved  
   → Sheet status `4 - Approved`  
   → **Mercury Log** gets `DRAFT_CREATED` (mock)
5. Select the Master Tracker row again  
6. **Invoice Prototype → Stacey: Pay selected Mercury draft**  
   → status `5 - Paid`, row moves to **Archive**, Mercury Log gets `PAYMENT_SETTLED`

### Also try
- **Inject TEST invoice (needs W-9)** → Waiting on Docs → **Simulate vendor submitted W-9** → then Ready → Approve/Reject  
- On a Ready row: **Simulate no-reply → On Hold**  
- On Approve email: click **REJECT** instead → Email Log notifies Stacey, **no** Mercury draft  
- **Run monthly archive review reminder**

---

## What “mock” looks like

### Email Log
Every automated email is a row:
- To / Subject / Body  
- Body includes real clickable Approve/Reject URLs after web app deploy  

Optional: in `Code.gs`, set `CONFIG.NOTIFY_EMAIL` to your address to also get real Gmail copies of the mocks.

### Mercury Log
- `DRAFT_CREATED` = pretend PDF forwarded to Mercury inbox  
- `PAYMENT_SETTLED` = pretend Stacey clicked Pay  

---

## Next (when you have Zapier / real email / Mercury)
Swap mocks one at a time:
1. Keep the Sheet as source of truth  
2. Replace Email Log writes with Gmail  
3. Replace Mercury Log draft with forward-to-Bill-Pay-inbox  
4. Replace Pay menu with Mercury settled webhook → Paid  

Until then, this Sheet **is** the working prototype of the full process.
