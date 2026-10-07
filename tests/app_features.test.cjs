const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(root, 'style.css'), 'utf8');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const server = fs.readFileSync(path.join(root, 'server.py'), 'utf8');

test('AI generation exposes an accessible progress bar with an indeterminate state', () => {
  assert.match(html, /id="aiProgress"[^>]*role="status"/);
  assert.match(html, /id="aiProgressTrack"/);
  assert.match(app, /setAiProgress\(message, progress=null\)/);
  assert.match(app, /aria-valuenow/);
  assert.match(css, /ai-progress-slide/);
});

test('voice references require a consent acknowledgement in the UI and local API', () => {
  assert.match(html, /id="aiVoiceConsent"/);
  assert.match(app, /usingReference&&!\$\('aiVoiceConsent'\)\.checked/);
  assert.match(app, /voiceConsent:\$\('aiVoiceConsent'\)\.checked/);
  assert.match(server, /request_data\.get\('voiceConsent'\) is not True/);
  assert.match(html, /특정 인물의 목소리를 그대로 복제/);
});

test('source bundles accept a folder or zip archive for parody voice analysis', () => {
  assert.match(html, /id="sourceBundleFile"/);
  assert.match(html, /id="sourceFolderFiles"[^>]*webkitdirectory/);
  assert.match(app, /sourceBundleFile/);
  assert.match(app, /handleSourceBundle/);
  assert.match(app, /handleSourceFolder/);
  assert.match(app, /qualityScore/);
  assert.match(server, /zipfile|\.zip/);
  assert.match(server, /MAX_ARCHIVE_UNPACKED_BYTES/);
  assert.match(server, /api\/source\//);
});
