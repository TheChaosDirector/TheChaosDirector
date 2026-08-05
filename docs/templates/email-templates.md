# Email templates

Copy/paste into Zapier Gmail / Email by Zapier steps. Replace `{{...}}` with Zapier fields.

---

## 1. W-9 / tax docs request (auto-reply to vendor)

**Subject:** Action needed — tax form for payments to {{Vendor}}

**Body:**

```text
Hi,

We received an invoice from you for {{Amount}} (Invoice {{Invoice Number}}).

Before we can process payment, we need a completed W-9 / tax form on file.

Please submit it here:
{{Typeform Link}}

Once we have it, we'll continue processing your invoice.

Thanks,
Finance
```

---

## 2. Head of School approval request

**Subject:** Invoice approval needed — {{Vendor}} — {{Amount}}

**Body (HTML-friendly plain text):**

```text
Hi,

An invoice is ready for your approval.

Vendor: {{Vendor}}
Invoice #: {{Invoice Number}}
Amount: {{Amount}}
Due date: {{Due Date}}
PDF: {{Drive Link}}

Please choose one:

APPROVE:
{{Approve Hook URL}}?decision=approve&row_id={{Row ID}}

REJECT:
{{Reject Hook URL}}?decision=reject&row_id={{Row ID}}

Thanks
```

Tip in Zapier: make APPROVE / REJECT look like buttons using an HTML email body with styled links.

---

## 3. Rejected notice to Stacey

**Subject:** Invoice rejected — {{Vendor}} — {{Amount}}

**Body:**

```text
Hi Stacey,

The Head of School rejected this invoice. It was NOT sent to Mercury.

Vendor: {{Vendor}}
Invoice #: {{Invoice Number}}
Amount: {{Amount}}
PDF: {{Drive Link}}
Rejected at: {{Rejected At}}
Reason: {{Reject Reason}}

Next step:
- Fix and set status back to "3 - Ready" (re-sends approval email), or
- Close as do-not-pay / move to Rejected archive

Master Tracker: {{Sheet Link}}
```

---

## 4. On Hold reminder (to Head + CC Stacey)

**Subject:** Reminder — invoice still waiting on approval — {{Vendor}}

**Body:**

```text
Hi,

This invoice is still waiting on an Approve / Reject decision (now On Hold).

Vendor: {{Vendor}}
Invoice #: {{Invoice Number}}
Amount: {{Amount}}
Due date: {{Due Date}}
PDF: {{Drive Link}}

APPROVE:
{{Approve Hook URL}}?decision=approve&row_id={{Row ID}}

REJECT:
{{Reject Hook URL}}?decision=reject&row_id={{Row ID}}

Stacey is CC'd.
```

---

## 5. Friday digest

**Subject:** Weekly invoice digest — week of {{Week Of}}

**Body:**

```text
Weekly invoice summary

Paid this week: {{Paid Count}} ({{Paid Total}})
Still Ready / waiting on Head: {{Ready Count}}
On Hold: {{On Hold Count}}
Rejected: {{Rejected Count}}
Waiting on Docs: {{Waiting Docs Count}}

Sheet: {{Sheet Link}}
```

---

## 6. Monthly / bi-monthly archive review reminder (to Stacey)

**Subject:** Archive review due — spot-check paid & rejected invoices

**Body:**

```text
Hi Stacey,

Time for the archive review.

Open the Archive tab:
{{Archive Sheet Link}}

Checklist:
1. Spot-check recent Paid rows (vendor, amount, date paid)
2. Review Rejected / closed items
3. Flag anything that looks wrong in Notes

When done, you can note today's date in Last Archive Review.
```
