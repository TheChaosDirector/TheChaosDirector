# HTML Approve / Reject email body

Use in Gmail "Send Email" with body type HTML (or Email by Zapier HTML).

Replace the `{{...}}` tokens with Zapier fields. Replace the hook base URLs with your Zap 5 Catch Hook URL.

```html
<div style="font-family: Arial, Helvetica, sans-serif; font-size: 15px; line-height: 1.5; color: #222;">
  <p>An invoice is ready for your approval.</p>

  <table style="border-collapse: collapse; margin: 16px 0;">
    <tr>
      <td style="padding: 4px 12px 4px 0; color: #555;">Vendor</td>
      <td style="padding: 4px 0;"><strong>{{Vendor}}</strong></td>
    </tr>
    <tr>
      <td style="padding: 4px 12px 4px 0; color: #555;">Invoice #</td>
      <td style="padding: 4px 0;">{{Invoice Number}}</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px 4px 0; color: #555;">Amount</td>
      <td style="padding: 4px 0;"><strong>{{Amount}}</strong></td>
    </tr>
    <tr>
      <td style="padding: 4px 12px 4px 0; color: #555;">Due date</td>
      <td style="padding: 4px 0;">{{Due Date}}</td>
    </tr>
  </table>

  <p><a href="{{Drive Link}}">View invoice PDF</a></p>

  <p style="margin: 24px 0 8px 0;">Please choose one:</p>

  <a href="{{Catch Hook URL}}?decision=approve&amp;row_id={{Row ID}}"
     style="display: inline-block; background: #1a7f37; color: #fff; text-decoration: none;
            padding: 12px 20px; border-radius: 6px; font-weight: bold; margin-right: 12px;">
    APPROVE
  </a>

  <a href="{{Catch Hook URL}}?decision=reject&amp;row_id={{Row ID}}"
     style="display: inline-block; background: #b42318; color: #fff; text-decoration: none;
            padding: 12px 20px; border-radius: 6px; font-weight: bold;">
    REJECT
  </a>

  <p style="margin-top: 24px; font-size: 12px; color: #666;">
    If the buttons don’t work, copy/paste these links:<br>
    Approve: {{Catch Hook URL}}?decision=approve&amp;row_id={{Row ID}}<br>
    Reject: {{Catch Hook URL}}?decision=reject&amp;row_id={{Row ID}}
  </p>
</div>
```

## Notes
- One Catch Hook can handle both decisions via the `decision` query param.
- Optional later upgrade: Reject link opens a short Typeform for a reject reason, then that form triggers Sheet update.
- Links are not secretly signed in this starter kit — treat Catch Hook URLs as sensitive and rotate if leaked.
