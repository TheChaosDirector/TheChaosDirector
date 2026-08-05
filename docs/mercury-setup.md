# Mercury Bill Pay setup notes

Zapier’s Mercury app does **not** create Bill Pay drafts. This workflow uses Mercury’s email inbox instead.

## What to set up in Mercury
1. Open Mercury → Payments / Bill Pay settings.
2. Copy the **Bill Pay forwarding email** (the address invoices get forwarded to).
3. Confirm who can review drafts and click Pay (Stacey).
4. Optional: set approval rules inside Mercury as a second control layer.

## What Zapier does on Approve
When Status becomes `4 - Approved`, Zap 5 should:
1. Get the invoice PDF (Drive link / file)
2. Email or forward that PDF **to the Mercury Bill Pay inbox address**
3. Use a clear subject, e.g. `Bill draft — {{Vendor}} — {{Invoice Number}} — {{Amount}}`

## What Stacey does
1. Open Mercury Bill Pay inbox
2. Confirm Mercury’s OCR draft matches the Sheet (vendor, amount, due date)
3. Click Pay / schedule payment

## What Zapier does after payment
Use Mercury **Settled Transaction** (Zap 7) to:
1. Match the Sheet row
2. Set Status `5 - Paid` + Date Paid
3. Move the row to Archive

## Matching tip
Include invoice # / vendor in the Mercury payment memo when possible so Zap 7 can find the row reliably.

## Do not
- Forward to Mercury on Reject, On Hold, or Waiting on Docs
- Auto-send money from Zapier (Stacey always clicks Pay)
