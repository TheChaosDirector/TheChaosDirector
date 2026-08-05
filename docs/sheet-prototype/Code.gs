/**
 * Invoice Approval Prototype — Google Apps Script
 *
 * Real: Google Sheet status machine, Approve/Reject web links, archive
 * Mock: Email (Email Log tab), Mercury Bill Pay (Mercury Log tab)
 *
 * Setup: see docs/sheet-prototype-setup.md
 */

const CONFIG = {
  MASTER: 'Master Tracker',
  VENDORS: 'Vendor Directory',
  ARCHIVE: 'Archive',
  EMAIL_LOG: 'Email Log',
  MERCURY_LOG: 'Mercury Log',
  // Optional: put your email here to ALSO get real Gmail notifications for the mock
  // Leave blank to only write to Email Log (no real mail required).
  NOTIFY_EMAIL: '',
};

const HEADERS = {
  master: [
    'Row ID', 'Date Received', 'Vendor', 'Invoice #', 'Amount', 'Due Date',
    'Drive Link', 'Status', 'Tax / W-9 OK?', 'Approved At', 'Rejected At',
    'Reject Reason', 'Date Paid', 'Last Archive Review', 'Notes', 'Approve Token',
  ],
  vendors: ['Vendor Name', 'Vendor Email', 'W-9 on File', 'W-9 Date', 'Typeform Response Link', 'Notes'],
  archive: [
    'Row ID', 'Date Received', 'Vendor', 'Invoice #', 'Amount', 'Due Date',
    'Drive Link', 'Final Status', 'Tax / W-9 OK?', 'Approved At', 'Rejected At',
    'Reject Reason', 'Date Paid', 'Archived At', 'Last Archive Review', 'Notes',
  ],
  emailLog: ['Timestamp', 'To', 'Subject', 'Body', 'Related Row ID', 'Mock?'],
  mercuryLog: ['Timestamp', 'Action', 'Vendor', 'Invoice #', 'Amount', 'Row ID', 'Status', 'Notes'],
};

const STATUS = {
  INTAKE: '1 - Intake',
  WAITING: '2 - Waiting on Docs',
  READY: '3 - Ready',
  APPROVED: '4 - Approved',
  PAID: '5 - Paid',
  REJECTED: '6 - Rejected',
  HOLD: '7 - On Hold',
};

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Invoice Prototype')
    .addItem('1) Setup sheet tabs', 'setupSheets')
    .addItem('2) Seed sample vendors', 'seedVendors')
    .addSeparator()
    .addItem('Inject TEST invoice (known vendor)', 'injectKnownVendorInvoice')
    .addItem('Inject TEST invoice (needs W-9)', 'injectNewVendorInvoice')
    .addSeparator()
    .addItem('Stacey: mark selected row Ready', 'staceyMarkReady')
    .addItem('Simulate vendor submitted W-9', 'simulateW9Submitted')
    .addItem('Simulate no-reply → On Hold', 'simulateOnHold')
    .addItem('Stacey: Pay selected Mercury draft', 'staceyPaySelected')
    .addItem('Run monthly archive review reminder', 'monthlyArchiveReviewReminder')
    .addSeparator()
    .addItem('Show Approve/Reject web app URL', 'showWebAppUrl')
    .addToUi();
}

function ss_() {
  const id = PropertiesService.getScriptProperties().getProperty('SPREADSHEET_ID');
  if (id) return SpreadsheetApp.openById(id);
  const active = SpreadsheetApp.getActiveSpreadsheet();
  if (active) {
    PropertiesService.getScriptProperties().setProperty('SPREADSHEET_ID', active.getId());
    return active;
  }
  throw new Error('Spreadsheet ID not saved. Run Invoice Prototype → 1) Setup sheet tabs from the Sheet first.');
}

/** Create tabs + headers if missing */
function setupSheets() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  PropertiesService.getScriptProperties().setProperty('SPREADSHEET_ID', ss.getId());
  ensureSheet_(ss, CONFIG.MASTER, HEADERS.master);
  ensureSheet_(ss, CONFIG.VENDORS, HEADERS.vendors);
  ensureSheet_(ss, CONFIG.ARCHIVE, HEADERS.archive);
  ensureSheet_(ss, CONFIG.EMAIL_LOG, HEADERS.emailLog);
  ensureSheet_(ss, CONFIG.MERCURY_LOG, HEADERS.mercuryLog);
  SpreadsheetApp.getUi().alert('Tabs ready: Master Tracker, Vendor Directory, Archive, Email Log, Mercury Log.');
}

function seedVendors() {
  setupSheets();
  const sheet = ss_().getSheetByName(CONFIG.VENDORS);
  if (sheet.getLastRow() <= 1) {
    sheet.appendRow(['Northwind Supplies LLC', 'ap@northwind.test', 'Yes', '2026-01-15', '', 'Demo known vendor']);
    sheet.appendRow(['Newco Contracting', 'billing@newco.test', 'No', '', '', 'Demo missing W-9']);
  }
  SpreadsheetApp.getUi().alert('Sample vendors seeded (if directory was empty).');
}

function injectKnownVendorInvoice() {
  setupSheets();
  seedVendorsIfEmpty_();
  const rowId = newRowId_();
  const vendor = 'Northwind Supplies LLC';
  createInvoiceRow_({
    rowId,
    vendor,
    invoiceNo: 'INV-' + Math.floor(1000 + Math.random() * 9000),
    amount: 1284.5,
    dueDate: daysFromNow_(15),
    taxOk: 'Yes',
    status: STATUS.INTAKE,
    notes: 'Prototype inject — known vendor',
  });
  logEmail_({
    to: 'finance@school.test',
    subject: 'MOCK inbox: invoice received from ' + vendor,
    body: 'Pretend this PDF hit the finance inbox. Row ' + rowId + ' created.',
    rowId,
  });
  // Skip waiting — jump Stacey-facing intake
  SpreadsheetApp.getUi().alert('Created ' + rowId + ' at status 1 - Intake.\nNext: select that row → Invoice Prototype → Stacey: mark Ready.');
}

function injectNewVendorInvoice() {
  setupSheets();
  seedVendorsIfEmpty_();
  const rowId = newRowId_();
  const vendor = 'Newco Contracting';
  createInvoiceRow_({
    rowId,
    vendor,
    invoiceNo: 'INV-' + Math.floor(1000 + Math.random() * 9000),
    amount: 640,
    dueDate: daysFromNow_(20),
    taxOk: 'No',
    status: STATUS.WAITING,
    notes: 'Prototype inject — waiting on W-9',
  });
  logEmail_({
    to: 'billing@newco.test',
    subject: 'MOCK auto-reply: please submit W-9',
    body: 'Hi Newco — submit tax docs: https://typeform.test/w9-demo\nRow: ' + rowId,
    rowId,
  });
  SpreadsheetApp.getUi().alert('Created ' + rowId + ' at 2 - Waiting on Docs.\nNext: Invoice Prototype → Simulate vendor submitted W-9 (select the row).');
}

function simulateW9Submitted() {
  const ctx = selectedMasterRow_();
  if (!ctx) return;
  const { sheet, row, data } = ctx;
  if (data.status !== STATUS.WAITING) {
    SpreadsheetApp.getUi().alert('Select a row with status "2 - Waiting on Docs".');
    return;
  }
  sheet.getRange(row, col_(HEADERS.master, 'Tax / W-9 OK?')).setValue('Yes');
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.INTAKE);
  sheet.getRange(row, col_(HEADERS.master, 'Notes')).setValue((data.notes || '') + ' | W-9 received (mock)');
  upsertVendorW9_(data.vendor);
  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK: W-9 received for ' + data.vendor,
    body: 'Tax docs in. Row ' + data.rowId + ' moved to Intake for review.',
    rowId: data.rowId,
  });
  SpreadsheetApp.getUi().alert('W-9 marked received. Status → 1 - Intake. Stacey can mark Ready.');
}

function staceyMarkReady() {
  const ctx = selectedMasterRow_();
  if (!ctx) return;
  const { sheet, row, data } = ctx;
  if (data.taxOk !== 'Yes') {
    SpreadsheetApp.getUi().alert('Tax / W-9 must be Yes before Ready.');
    return;
  }
  const token = data.token || Utilities.getUuid();
  sheet.getRange(row, col_(HEADERS.master, 'Approve Token')).setValue(token);
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.READY);

  const base = ScriptApp.getService().getUrl();
  const approveUrl = base ? base + '?decision=approve&row_id=' + encodeURIComponent(data.rowId) + '&token=' + encodeURIComponent(token) : '(Deploy web app first — use menu: Show Approve/Reject web app URL)';
  const rejectUrl = base ? base + '?decision=reject&row_id=' + encodeURIComponent(data.rowId) + '&token=' + encodeURIComponent(token) : '(Deploy web app first)';

  const body =
    'Invoice approval needed\n\n' +
    'Vendor: ' + data.vendor + '\n' +
    'Invoice #: ' + data.invoiceNo + '\n' +
    'Amount: $' + data.amount + '\n' +
    'Due: ' + data.dueDate + '\n' +
    'PDF: ' + (data.driveLink || '(mock drive link)') + '\n\n' +
    'APPROVE:\n' + approveUrl + '\n\n' +
    'REJECT:\n' + rejectUrl + '\n';

  logEmail_({
    to: 'head@school.test',
    subject: 'MOCK approval email — ' + data.vendor + ' — $' + data.amount,
    body,
    rowId: data.rowId,
  });

  SpreadsheetApp.getUi().alert(
    'Status → 3 - Ready.\nMock approval email written to Email Log.\n\n' +
    'Open Email Log and click APPROVE or REJECT link.\n' +
    '(If links say deploy first: Extensions → Apps Script → Deploy → New deployment → Web app.)'
  );
}

function simulateOnHold() {
  const ctx = selectedMasterRow_();
  if (!ctx) return;
  const { sheet, row, data } = ctx;
  if (data.status !== STATUS.READY) {
    SpreadsheetApp.getUi().alert('Select a row at 3 - Ready.');
    return;
  }
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.HOLD);
  logEmail_({
    to: 'head@school.test',
    subject: 'MOCK reminder — still waiting on approval — ' + data.vendor,
    body: 'Row ' + data.rowId + ' moved to 7 - On Hold. Stacey CC’d (mock).',
    rowId: data.rowId,
  });
  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK: invoice On Hold — ' + data.rowId,
    body: 'Head has not responded. Status 7 - On Hold.',
    rowId: data.rowId,
  });
  SpreadsheetApp.getUi().alert('Status → 7 - On Hold. Reminders logged in Email Log.');
}

/** Web app entry — Approve / Reject links land here */
function doGet(e) {
  const decision = (e.parameter.decision || '').toLowerCase();
  const rowId = e.parameter.row_id || '';
  const token = e.parameter.token || '';
  try {
    if (decision === 'approve') {
      approveRow_(rowId, token);
      return htmlPage_('Approved', 'Row ' + rowId + ' set to 4 - Approved. Mock Mercury draft created. Stacey can Pay from the Sheet menu.');
    }
    if (decision === 'reject') {
      rejectRow_(rowId, token, e.parameter.reason || 'Rejected via prototype link');
      return htmlPage_('Rejected', 'Row ' + rowId + ' set to 6 - Rejected. Nothing sent to Mercury. Stacey was notified in Email Log.');
    }
    return htmlPage_('Invoice prototype', 'Use Approve/Reject links from the Email Log.');
  } catch (err) {
    return htmlPage_('Error', String(err));
  }
}

function approveRow_(rowId, token) {
  const found = findMasterByRowId_(rowId);
  if (!found) throw new Error('Row not found: ' + rowId);
  const { sheet, row, data } = found;
  if (data.token && token && data.token !== token) throw new Error('Invalid token');
  if (data.status !== STATUS.READY && data.status !== STATUS.HOLD) {
    throw new Error('Row must be Ready or On Hold to approve. Current: ' + data.status);
  }
  const now = new Date();
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.APPROVED);
  sheet.getRange(row, col_(HEADERS.master, 'Approved At')).setValue(now);

  // MOCK Mercury: log draft instead of real Bill Pay
  const m = ss_().getSheetByName(CONFIG.MERCURY_LOG);
  m.appendRow([
    now, 'DRAFT_CREATED', data.vendor, data.invoiceNo, data.amount, data.rowId, 'Draft (mock)',
    'Pretend PDF was forwarded to Mercury Bill Pay inbox',
  ]);

  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK: approved — Mercury draft ready — ' + data.rowId,
    body: 'Head approved. Mock Mercury draft is in Mercury Log. Use menu: Stacey: Pay selected Mercury draft (select Master Tracker row).',
    rowId: data.rowId,
  });
}

function rejectRow_(rowId, token, reason) {
  const found = findMasterByRowId_(rowId);
  if (!found) throw new Error('Row not found: ' + rowId);
  const { sheet, row, data } = found;
  if (data.token && token && data.token !== token) throw new Error('Invalid token');
  const now = new Date();
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.REJECTED);
  sheet.getRange(row, col_(HEADERS.master, 'Rejected At')).setValue(now);
  sheet.getRange(row, col_(HEADERS.master, 'Reject Reason')).setValue(reason || '');

  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK: invoice rejected — ' + data.rowId,
    body: 'Rejected. NOT sent to Mercury.\nReason: ' + (reason || '') + '\nFix & mark Ready, or close out.',
    rowId: data.rowId,
  });
}

function staceyPaySelected() {
  const ctx = selectedMasterRow_();
  if (!ctx) return;
  const { sheet, row, data } = ctx;
  if (data.status !== STATUS.APPROVED) {
    SpreadsheetApp.getUi().alert('Select a row at 4 - Approved (after Head approve + mock Mercury draft).');
    return;
  }
  const now = new Date();
  sheet.getRange(row, col_(HEADERS.master, 'Status')).setValue(STATUS.PAID);
  sheet.getRange(row, col_(HEADERS.master, 'Date Paid')).setValue(now);

  const m = ss_().getSheetByName(CONFIG.MERCURY_LOG);
  m.appendRow([now, 'PAYMENT_SETTLED', data.vendor, data.invoiceNo, data.amount, data.rowId, 'Paid (mock)', 'Stacey clicked Pay (prototype)']);

  archiveRow_(sheet, row);
  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK Friday digest item — paid ' + data.rowId,
    body: 'Paid $' + data.amount + ' to ' + data.vendor + '. Row archived.',
    rowId: data.rowId,
  });
  SpreadsheetApp.getUi().alert('Paid (mock) + archived. Check Archive + Mercury Log + Email Log.');
}

function monthlyArchiveReviewReminder() {
  setupSheets();
  logEmail_({
    to: 'stacey@school.test',
    subject: 'MOCK: monthly/bi-monthly archive review due',
    body: 'Open the Archive tab and spot-check paid/rejected rows. Flag anything odd in Notes.',
    rowId: '',
  });
  SpreadsheetApp.getUi().alert('Archive review reminder written to Email Log.');
}

function showWebAppUrl() {
  const url = ScriptApp.getService().getUrl();
  SpreadsheetApp.getUi().alert(url
    ? ('Web app URL:\n\n' + url + '\n\nApprove/Reject links in Email Log use this.')
    : 'Not deployed yet.\n\nExtensions → Apps Script → Deploy → New deployment\nType: Web app\nExecute as: Me\nWho has access: Anyone (for prototype links)\nThen re-run Stacey: mark Ready.');
}

/* ================= helpers ================= */

function ensureSheet_(ss, name, headers) {
  let sheet = ss.getSheetByName(name);
  if (!sheet) sheet = ss.insertSheet(name);
  const existing = sheet.getRange(1, 1, 1, headers.length).getValues()[0];
  const blank = existing.every((c) => c === '');
  if (blank || sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function seedVendorsIfEmpty_() {
  const sheet = ss_().getSheetByName(CONFIG.VENDORS);
  if (sheet.getLastRow() <= 1) seedVendors();
}

function createInvoiceRow_(p) {
  const sheet = ss_().getSheetByName(CONFIG.MASTER);
  const token = Utilities.getUuid();
  sheet.appendRow([
    p.rowId,
    new Date(),
    p.vendor,
    p.invoiceNo,
    p.amount,
    p.dueDate,
    'https://drive.google.com/file/d/mock-' + p.rowId,
    p.status,
    p.taxOk,
    '', '', '', '', '',
    p.notes || '',
    token,
  ]);
}

function archiveRow_(masterSheet, row) {
  const values = masterSheet.getRange(row, 1, 1, HEADERS.master.length).getValues()[0];
  const archive = ss_().getSheetByName(CONFIG.ARCHIVE);
  // Map master → archive (drop Approve Token; add Archived At)
  archive.appendRow([
    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
    values[7], values[8], values[9], values[10], values[11], values[12],
    new Date(), values[13], values[14],
  ]);
  masterSheet.deleteRow(row);
}

function logEmail_(p) {
  const sheet = ss_().getSheetByName(CONFIG.EMAIL_LOG);
  sheet.appendRow([new Date(), p.to, p.subject, p.body, p.rowId || '', 'YES — MOCK']);
  if (CONFIG.NOTIFY_EMAIL) {
    try {
      MailApp.sendEmail(CONFIG.NOTIFY_EMAIL, '[PROTOTYPE] ' + p.subject, p.body);
    } catch (err) {
      // ignore mail failures in prototype
    }
  }
}

function upsertVendorW9_(vendorName) {
  const sheet = ss_().getSheetByName(CONFIG.VENDORS);
  const data = sheet.getDataRange().getValues();
  for (let i = 1; i < data.length; i++) {
    if (String(data[i][0]).toLowerCase() === String(vendorName).toLowerCase()) {
      sheet.getRange(i + 1, 3).setValue('Yes');
      sheet.getRange(i + 1, 4).setValue(new Date());
      return;
    }
  }
  sheet.appendRow([vendorName, '', 'Yes', new Date(), '', 'Added by prototype']);
}

function selectedMasterRow_() {
  const sheet = ss_().getSheetByName(CONFIG.MASTER);
  const range = SpreadsheetApp.getActive().getActiveRange();
  if (!range || range.getSheet().getName() !== CONFIG.MASTER) {
    SpreadsheetApp.getUi().alert('Click a cell on a data row in Master Tracker first.');
    return null;
  }
  const row = range.getRow();
  if (row < 2) {
    SpreadsheetApp.getUi().alert('Select a data row (not the header).');
    return null;
  }
  return { sheet, row, data: readMasterRow_(sheet, row) };
}

function findMasterByRowId_(rowId) {
  const sheet = ss_().getSheetByName(CONFIG.MASTER);
  const data = sheet.getDataRange().getValues();
  for (let i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(rowId)) {
      return { sheet, row: i + 1, data: readMasterRow_(sheet, i + 1) };
    }
  }
  return null;
}

function readMasterRow_(sheet, row) {
  const v = sheet.getRange(row, 1, 1, HEADERS.master.length).getValues()[0];
  return {
    rowId: v[0],
    vendor: v[2],
    invoiceNo: v[3],
    amount: v[4],
    dueDate: v[5],
    driveLink: v[6],
    status: v[7],
    taxOk: v[8],
    notes: v[14],
    token: v[15],
  };
}

function col_(headers, name) {
  const i = headers.indexOf(name);
  if (i < 0) throw new Error('Missing header ' + name);
  return i + 1;
}

function newRowId_() {
  return 'INV-DEMO-' + Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyyMMdd-HHmmss');
}

function daysFromNow_(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return Utilities.formatDate(d, Session.getScriptTimeZone(), 'yyyy-MM-dd');
}

function htmlPage_(title, msg) {
  const html = HtmlService.createHtmlOutput(
    '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"/>' +
    '<body style="font-family:system-ui,sans-serif;padding:32px;max-width:560px;line-height:1.45">' +
    '<h1 style="margin:0 0 12px;font-size:22px">' + title + '</h1>' +
    '<p style="margin:0;color:#333">' + msg + '</p>' +
    '<p style="margin-top:24px;color:#666;font-size:13px">You can close this tab and return to the Google Sheet.</p>' +
    '</body>'
  );
  html.setTitle(title);
  return html;
}
