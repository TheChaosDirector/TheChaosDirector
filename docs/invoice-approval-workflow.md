# Vendor Invoice Approval Workflow

Plain-English process for finance intake → approval (or rejection) → Mercury payment.

**People involved**
- **Automation (Zapier):** filing, emails, status updates, Mercury forward
- **Stacey:** compliance check + final Pay in Mercury (and handling rejects)
- **Head of School:** Approve or Reject

---

## Statuses used in the Sheet

| Status | Meaning | Goes to Mercury? |
|---|---|---|
| `1 - Intake` | New invoice landed; being processed | No |
| `2 - Waiting on Docs` | Vendor needs to send W-9 / tax form | No |
| `3 - Ready` | Stacey reviewed it; waiting on Head of School | No |
| `4 - Approved` | Head approved; PDF forwarded to Mercury | Yes |
| `5 - Paid` | Mercury payment settled | Already there |
| `6 - Rejected` | Head rejected; Stacey must fix or close | No |
| `7 - On Hold` | Head never responded; needs a follow-up | No |

**Hard rule:** only status **`4 - Approved`** triggers the Mercury Bill Pay forward.

---

## One diagram (approve, reject, and no reply)

```mermaid
flowchart TD
  A[Invoice arrives in finance email] --> B{Vendor known<br/>and W-9 on file?}

  B -->|No| C[Auto-reply: W-9 / Typeform link]
  C --> D[Status: 2 - Waiting on Docs]
  D --> E[Vendor submits tax docs]
  E --> F[AI reads invoice PDF]

  B -->|Yes| F

  F --> G[Save PDF to Drive<br/>Create Master Tracker row]
  G --> H[Stacey reviews row]
  H --> I[Status: 3 - Ready]
  I --> J[Email Head of School<br/>PDF + Approve + Reject]

  J --> K{Head of School decision}

  K -->|Approve| L[Status: 4 - Approved<br/>+ timestamp]
  L --> M[Forward PDF to Mercury Bill Pay inbox]
  M --> N[Mercury builds draft bill]
  N --> O[Stacey reviews and clicks Pay]
  O --> P[Mercury payment settles]
  P --> Q[Status: 5 - Paid<br/>Date Paid logged]
  Q --> R[Archive row<br/>Include in Friday summary]

  K -->|Reject| S[Status: 6 - Rejected<br/>+ timestamp + reason]
  S --> T[Email Stacey — do not send to Mercury]
  T --> U{Can it be fixed?}
  U -->|Yes| H
  U -->|No| V[Close / Rejected archive]

  K -->|No reply in 3 business days| W[Status: 7 - On Hold]
  W --> X[Remind Head + notify Stacey]
  X --> J
```

---

## Who does what (including not approved)

```text
INVOICE EMAIL
      │
      ▼
┌──────────────────────────────────────┐
│ AUTOMATION                           │
│ • Checks vendor / W-9                │
│ • Sends tax form if needed           │
│ • Reads invoice with AI              │
│ • Saves PDF + creates Sheet row      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ STACEY                               │
│ • Quick Sheet review                 │
│ • Sets status to "3 - Ready"         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ HEAD OF SCHOOL                       │
│ Email has two buttons:               │
│   [ APPROVE ]     [ REJECT ]         │
└───────┬──────────────────┬───────────┘
        │                  │
 Approve│                  │Reject
        ▼                  ▼
┌─────────────────┐  ┌─────────────────────────────┐
│ AUTOMATION      │  │ AUTOMATION                  │
│ Status:         │  │ Status: 6 - Rejected        │
│ 4 - Approved    │  │ Email Stacey with reason    │
│ Forward PDF to  │  │ Do NOT send to Mercury      │
│ Mercury inbox   │  └──────────────┬──────────────┘
└────────┬────────┘                 │
         │                          ▼
         ▼               ┌─────────────────────────────┐
┌─────────────────┐      │ STACEY                      │
│ MERCURY         │      │ • Fix + set Ready again     │
│ Builds draft    │      │   (re-sends approval email) │
└────────┬────────┘      │ • Or close as do-not-pay    │
         │               └─────────────────────────────┘
         ▼
┌─────────────────┐
│ STACEY          │
│ Checks draft    │
│ Clicks Pay      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AUTOMATION      │
│ Marks Paid      │
│ Archives row    │
│ Friday summary  │
└─────────────────┘

If Head never clicks either button:
  Automation → Status 7 - On Hold → reminder to Head + notify Stacey
```

---

## What happens if it is NOT approved

### Head clicks Reject
1. Sheet → **`6 - Rejected`** (timestamp + optional reason).
2. Stacey gets an email with vendor, amount, invoice #, PDF link, and reason.
3. **No Mercury draft.** Nothing is forwarded.
4. Stacey either:
   - **Fixes it** (corrected invoice / missing info) → set status back to **`3 - Ready`** → Head gets a new approval email, or
   - **Closes it** as do-not-pay → leave Rejected / move to Rejected archive.

### Head never clicks anything
1. After **3 business days** → status **`7 - On Hold`**.
2. Reminder email to Head + notify Stacey.
3. Stacey can nudge, wait, or pull the invoice back.

---

## Suggested Sheet columns

| Column | Purpose |
|---|---|
| A – Date Received | When email landed |
| B – Vendor | From AI / email |
| C – Invoice # | From AI |
| D – Amount | From AI |
| E – Due Date | From AI |
| F – Drive Link | PDF in Google Drive |
| G – Status | 1–7 status values above |
| H – Tax / W-9 OK? | Yes / No |
| I – Approved At | Timestamp when Head approved |
| J – Rejected At | Timestamp when Head rejected |
| K – Reject Reason | Optional note from reject link |
| L – Date Paid | From Mercury settlement |
| M – Row ID | Stable ID used in Approve/Reject links |

---

## Zapier build (separate Zaps, reject included)

### Zap 1 — Intake
- **Trigger:** New email in finance inbox (PDF attachment)
- **Lookup** vendor in Sheet / vendor directory
- **Path A — missing tax:** reply with Typeform/W-9 link → create/update row → status `2`
- **Path B — tax OK:** continue to parse

### Zap 2 — Tax form received
- **Trigger:** Typeform submission
- Mark vendor W-9 complete
- If a waiting invoice exists for that vendor → move it forward to parse / notify Stacey

### Zap 3 — Parse + file
- AI extract: vendor, invoice #, amount, due date
- Upload PDF to Drive
- Create Sheet row (status `1` or ready for Stacey review)
- Deduplicate on vendor + invoice #

### Zap 4 — Ready → ask Head
- **Trigger:** Sheet status becomes `3 - Ready`
- Email Head of School with:
  - Vendor, amount, invoice #, due date, Drive PDF link
  - **Approve** link → Catch Hook `?decision=approve&row_id=...`
  - **Reject** link → Catch Hook `?decision=reject&row_id=...` (optional reason field / separate reject hook)

### Zap 5 — Decision webhook (Approve or Reject)
- **Trigger:** Webhooks Catch Hook
- **If Approve:**
  - Update row → `4 - Approved` + Approved At
  - Email/forward the PDF to Mercury Bill Pay inbox
- **If Reject:**
  - Update row → `6 - Rejected` + Rejected At (+ reason)
  - Email Stacey
  - **Do not** contact Mercury

### Zap 6 — No response / On Hold
- **Trigger:** Schedule (daily)
- Find rows still at `3 - Ready` older than 3 business days
- Set `7 - On Hold`
- Remind Head + notify Stacey

### Zap 7 — Paid
- **Trigger:** Mercury Settled Transaction
- Match Sheet row (amount / vendor / memo / invoice #)
- Status `5 - Paid` + Date Paid
- Move to Archive tab

### Zap 8 — Friday digest
- **Trigger:** Every Friday
- Email summary of Paid / Rejected / On Hold / still Ready this week

---

## Bottom line for Stacey

**Normal invoice**
1. Review the Sheet row → set **Ready**
2. Later, review the Mercury draft → click **Pay**

**If the Head rejects**
- You get an email, not a Mercury draft
- Fix and set Ready again, or close it as do-not-pay

**If the Head goes silent**
- Row goes On Hold and you get a ping so it doesn’t disappear
