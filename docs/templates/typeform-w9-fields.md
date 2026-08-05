# Typeform — W-9 / tax docs form

Create a Typeform (or Google Form) with these fields. Point Zap 2 at new submissions.

## Fields

| Field | Type | Required | Maps to |
|---|---|---|---|
| Vendor / Company Name | Short text | Yes | Vendor Directory → Vendor Name |
| Contact Email | Email | Yes | Vendor Directory → Vendor Email |
| Invoice Number (if known) | Short text | No | Used to find waiting Sheet rows |
| W-9 / tax document upload | File upload | Yes | Store Drive link or keep Typeform file URL |
| Tax classification (optional) | Multiple choice | No | Individual / LLC / Corp / Other |
| Notes | Long text | No | Vendor Directory → Notes |

## Ending screen copy

```text
Thanks — we received your tax docs.
We’ll continue processing your invoice shortly.
```

## Zap 2 mapping reminder
1. Update/create Vendor Directory row: `W-9 on File = Yes`, `W-9 Date = today`
2. Save Typeform response / file link
3. Search Master Tracker for that vendor with Status `2 - Waiting on Docs`
4. Notify Stacey that docs arrived (optional: auto-move to parse if invoice PDF already filed)
