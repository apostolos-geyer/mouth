/**
 * Tests for the pi extension. Needs a TypeScript-aware runtime:
 *
 *     bun extensions/mouth.test.mjs
 *
 * Not part of `pytest` -- it is the only JavaScript in the repo and the only thing that
 * needs bun, so it stays opt-in rather than making the Python suite depend on a
 * second toolchain.
 *
 * The extension is driven for real: a stand-in for `m dictate --hold --events` replays a
 * scripted event stream on stderr, and a fake pi records what lands in the editor and the
 * footer. That is the whole contract -- everything the extension does is a reaction to
 * those events, so scripting them exercises the actual code rather than a paraphrase.
 */

import { mkdtempSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ---- a stand-in for `m dictate --hold --events` --------------------------------

const dir = mkdtempSync(join(tmpdir(), "mouth-ext-"));
writeFileSync(
	join(dir, "fake-m.mjs"),
	`const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (const e of JSON.parse(process.env.SCRIPT)) {
  process.stderr.write(JSON.stringify(e) + "\\n");
  await sleep(12);
}
// --hold is what makes it a mode: it does not exit on its own, it waits for the signal.
await new Promise(() => {});
`,
);
writeFileSync(join(dir, "m"), `#!/bin/sh\nexec ${process.execPath} ${dir}/fake-m.mjs "$@"\n`);
chmodSync(join(dir, "m"), 0o755);
process.env.MOUTH_BIN = join(dir, "m");

// Imported here, not at the top: static imports are hoisted, so the module would read
// MOUTH_BIN before the line above ran and shell out to a real `m`.
const { default: mouthExtension } = await import("./mouth.ts");

// ---- a fake pi ------------------------------------------------------------------

let editor = "";
let status = [];
const ctx = {
	hasUI: true,
	ui: {
		getEditorText: () => editor,
		setEditorText: (t) => {
			editor = t;
		},
		setStatus: (_key, text) => {
			if (text !== undefined) status.push(text);
		},
		notify: () => {},
	},
};

let toggle;
mouthExtension({ registerCommand: (_n, o) => (toggle = o.handler), on: () => {} });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const P = (text) => ({ event: "partial", text });
const F = (text) => ({ event: "final", text });
const READY = { event: "ready", threshold: 0.005, calibrated: true, load: 0.3 };
const LEVELS = Array.from({ length: 12 }, () => ({ event: "level", rms: 0.02, speech: true }));

/** Start the listener on `script`, optionally reach in and change the editor, then stop. */
async function session(script, { editorBecomes, at = 40 } = {}) {
	editor = "";
	status = [];
	process.env.SCRIPT = JSON.stringify(script);
	await toggle("", ctx);
	if (editorBecomes !== undefined) {
		await sleep(at);
		editor = editorBecomes;
	}
	await sleep(400);
	await toggle("", ctx);
}

let failed = 0;
function check(name, got, want) {
	const ok = JSON.stringify(got) === JSON.stringify(want);
	if (!ok) failed++;
	console.log(`  ${ok ? "ok  " : "FAIL"} ${name}`);
	if (!ok) console.log(`         got:  ${JSON.stringify(got)}\n         want: ${JSON.stringify(want)}`);
}

// A final refines the partials before it rather than landing after them, or every
// sentence would appear twice: once forming, once finished.
await session([READY, ...LEVELS, P("Hello"), P("Hello there"), F("Hello there.")]);
check("partials refine into one final", editor, "Hello there.");

await session([READY, P("Hello"), F("Hello there."), P("How are"), F("How are you?")]);
check("a second utterance appends", editor, "Hello there. How are you?");

// Words belong in one place. They used to be in the editor *and* quoted in the footer,
// which is the same sentence twice on one screen.
await session([READY, ...LEVELS, P("Hello there"), F("Hello there.")]);
check("footer carries no transcript", status.some((s) => /Hello/.test(s)), false);
check("footer carries a meter", status.some((s) => /[█░]/.test(s)), true);

// Enter is inferred, not subscribed to: pi clears the editor, so the next write sees
// contents that are not what it last wrote and rebases instead of re-appending.
await session([READY, F("First one."), P("Second"), F("Second one.")], { editorBecomes: "" });
check("Enter rebases rather than re-appending", editor, "Second one.");

await session([READY, F("Spoken."), P("more"), F("more spoken.")], {
	editorBecomes: "typed by hand",
});
check("text typed by hand is not eaten", editor, "typed by hand more spoken.");

// `close` lands well after the kill that caused it. A stop-then-start inside that window
// had the dying child's handler shut down its replacement, and the next thing you said
// went nowhere while the footer still read "listening".
{
	editor = "";
	process.env.SCRIPT = JSON.stringify([READY, F("first.")]);
	await toggle("", ctx);
	await sleep(120);
	await toggle("", ctx);
	process.env.SCRIPT = JSON.stringify([READY, P("second"), F("second one.")]);
	await toggle("", ctx);
	await sleep(400);
	// Stopping does not clear the editor -- that would throw away what you dictated -- so
	// restarting continues from whatever is in the box.
	check("a restart survives the old child's close", editor, "first. second one.");
	await toggle("", ctx);
}

console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
