# Invoice Payment Flow — For Stacey

**[Open the click-through demo](https://htmlpreview.github.io/?https://github.com/TheChaosDirector/TheChaosDirector/blob/cursor/invoice-workflow-reject-path-8a76/docs/demo/invoice-flow-prototype.html)** (email + Mercury are mocked)

Share this page or the image below.

![Invoice approval workflow](./invoice-workflow-diagram.png)

**Download:** [PNG](./invoice-workflow-diagram.png) · [SVG](./invoice-workflow-diagram.svg)

---

## Your two main jobs

1. **Compliance check** — When a new row lands in the Master Tracker, review contract / tax / numbers, then set status to **`3 - Ready`**.
2. **Pay in Mercury** — After the Head of School approves, open the Mercury draft, sanity-check it, and click **Pay**.

---

## If the Head rejects

- You get an email (nothing goes to Mercury).
- Fix the issue and set status back to **`3 - Ready`**, or close it as do-not-pay.

## If the Head doesn’t respond

- After ~3 business days the row goes **`7 - On Hold`** and you get a ping.
- Nudge the Head, or pull it back if it should wait.

## Once a month (or twice a month)

- You’ll get a reminder to open the **Archive** tab.
- Spot-check paid and rejected items. Flag anything that looks wrong.

---

## Status cheat sheet

| Status | Meaning |
|---|---|
| 2 - Waiting on Docs | Vendor still needs W-9 / tax form |
| 3 - Ready | Waiting on Head of School |
| 4 - Approved | Sent to Mercury Bill Pay inbox |
| 5 - Paid | Payment settled |
| 6 - Rejected | Head said no — fix or close |
| 7 - On Hold | Head hasn’t replied |

**Only Approved invoices go to Mercury.**
