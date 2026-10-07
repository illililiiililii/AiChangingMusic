const test = require('node:test');
const assert = require('node:assert/strict');
const { structureLyricsForAI, compareLyricText } = require('../lyrics_structure.js');

const sample = [
  'Walking past the station',
  'Morning lights are fading',
  'I can hear the city',
  'We will sing together',
  'Keep the bright lights burning',
  'Carry every note',
  'A quiet road behind us',
  'We will sing together',
  'Keep the bright lights burning',
  'Carry every note',
  'Take my hand tonight',
  'We can make it home',
].join('\n');

test('finds a repeated chorus and preserves every lyric line in order', () => {
  const output = structureLyricsForAI(sample);
  assert.match(output, /\[Verse 1\]/);
  assert.equal((output.match(/\[Chorus\]/g) || []).length, 2);
  const lyricsOnly = output.split(/\r?\n/).filter((line) => !/^\[.*\]$/.test(line));
  assert.deepEqual(lyricsOnly, sample.split('\n'));
});

test('leaves supplied section tags and disabled auto-structure unchanged', () => {
  const manual = '[Verse 1]\nOne short line\n[Chorus]\nA repeated hook';
  assert.equal(structureLyricsForAI(manual), manual);
  assert.equal(structureLyricsForAI(sample, false), sample);
});

test('scores recognized lyrics and rejects unrelated transcript text', () => {
  assert.equal(compareLyricText('오늘은 노래해', '오늘은 노래해'), 1);
  assert.ok(compareLyricText('오늘은 노래해', '오늘 노래해') > 0.6);
  assert.equal(compareLyricText('오늘은 노래해', '댄스의 댄스를 포함하고 있습니다'), 0);
});
