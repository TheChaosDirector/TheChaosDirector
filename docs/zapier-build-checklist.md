# Zapier build checklist

Use this when standing up the live Zaps. Keep Zaps separate (one trigger → one outcome).

**Prereqs**
- [ ] Finance Gmail/Outlook inbox ready
- [ ] Google Sheet created from `templates/master-tracker-headers.csv`
- [ ] Vendor Directory tab/sheet from `templates/vendor-directory-headers.csv`
- [ ] Google Drive folder for invoice PDFs
- [ ] Typeform (or Google Form) for W-9 / tax docs
- [ ] Mercury Bill Pay inbox email address copied
- [ ] Head of School + Stacey email addresses confirmed
- [ ] Zapier account connected to Gmail, Sheets, Drive, Mercury, Typeform

---

## Zap 1 — Intake + vendor check
- [ ] Trigger: New email in finance inbox (filter: has PDF / subject contains invoice)
- [ ] Lookup vendor in Vendor Directory
- [ ] Path A — missing W-9: send Email template 1 → create/update row Status `2 - Waiting on Docs`
- [ ] Path B — W-9 OK: pass attachment to Zap 3 (or continue in Paths)

## Zap 2 — Tax form received
- [ ] Trigger: Typeform new entry
- [ ] Update Vendor Directory: W-9 on File = Yes + date
- [ ] Find waiting invoice rows for that vendor
- [ ] Notify Stacey / move those rows forward

## Zap 3 — Parse + file
- [ ] AI extract: Vendor, Invoice #, Amount, Due Date
- [ ] Upload PDF to Drive folder
- [ ] Deduplicate: Search Sheets for Vendor + Invoice #
- [ ] Create row with Row ID + Status for Stacey review
- [ ] Stop if duplicate found (optional: notify Stacey)

## Zap 4 — Ready → ask Head
- [ ] Trigger: New/Updated Sheet row where Status = `3 - Ready`
- [ ] Send Email template 2 to Head of School
- [ ] Include Approve + Reject Catch Hook URLs with `row_id`

## Zap 5 — Decision webhook
- [ ] Trigger: Webhooks by Zapier — Catch Hook
- [ ] Branch on `decision`
- [ ] Approve path:
  - [ ] Update row → `4 - Approved` + Approved At
  - [ ] Email/forward PDF to Mercury Bill Pay inbox
- [ ] Reject path:
  - [ ] Update row → `6 - Rejected` + Rejected At (+ reason)
  - [ ] Send Email template 3 to Stacey
  - [ ] Do **not** contact Mercury

## Zap 6 — On Hold
- [ ] Trigger: Schedule daily
- [ ] Find rows Status = `3 - Ready` older than 3 business days
- [ ] Set Status `7 - On Hold`
- [ ] Send Email template 4 to Head (CC Stacey)

## Zap 7 — Paid
- [ ] Trigger: Mercury Settled Transaction
- [ ] Lookup matching Sheet row (amount / vendor / memo / invoice #)
- [ ] Update Status `5 - Paid` + Date Paid
- [ ] Move row to Archive tab (or copy + delete)

## Zap 8 — Friday digest
- [ ] Trigger: Every Friday
- [ ] Count/summarize statuses
- [ ] Send Email template 5 to Stacey (+ optional Head)

## Zap 9 — Archive review reminder
- [ ] Trigger: 1st of month (or 1st + 15th)
- [ ] Send Email template 6 to Stacey
- [ ] Link to Archive tab

---

## Go-live smoke test
- [ ] Send a test invoice PDF from a known vendor with W-9 → row created
- [ ] Stacey sets Ready → Head gets email
- [ ] Click Reject → Stacey notified, Mercury not contacted
- [ ] Set Ready again → Approve → PDF lands in Mercury Bill Pay
- [ ] Mark a test payment settled (or simulate) → Status Paid + archived
- [ ] Confirm Friday digest + archive reminder schedules are ON

## Hard rules
1. Only Status `4 - Approved` forwards to Mercury.
2. Rejected / On Hold / Waiting on Docs never create Mercury drafts.
3. Humans remain at: Stacey compliance, Head Approve/Reject, Stacey Pay, monthly archive review.
