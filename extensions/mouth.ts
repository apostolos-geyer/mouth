/**
 * mouth — speak into pi's editor.
 *
 * One command, `/mouth`, which toggles a listener that stays up:
 *
 *   - every pause commits a sentence into the editor
 *   - Enter sends, and the listener survives it
 *   - `/mouth` again stops it
 *
 * It shells out to `m dictate --hold --events`, which emits newline-delimited JSON on
 * stderr while keeping stdout clean. Levels drive a meter in the footer, partials show
 * the sentence forming, finals land in the editor.
 *
 * `/mouth install` installs the CLI this drives, straight from the repo, so nothing here
 * depends on where anyone keeps a checkout. Set MOUTH_BIN to use a different binary.
 * Nothing is bundled: transcription is local, and this file only marshals between one
 * process and one text box.
 */

import type {
	ExtensionAPI,
	ExtensionCommandContext,
} from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcess } from "node:child_process";

const M_BIN = process.env.MOUTH_BIN ?? "m";

/** Where `/mouth install` installs from. MOUTH_REPO points it at a fork or a local path. */
const REPO = process.env.MOUTH_REPO ?? "git+https://github.com/apostolos-geyer/mouth";

/**
 * The optional dependency groups worth having here, from the platform table in the repo's
 * README: MLX ships arm64-macOS wheels only, and diarization is CoreML, so it is macOS at
 * all and slow off Apple Silicon. Asking for an extra that cannot resolve fails the whole
 * install, so the ones that cannot work are simply not requested.
 */
function extrasForThisMachine(): string {
	if (process.platform !== "darwin") return "";
	return process.arch === "arm64" ? "[mlx,diarize]" : "[diarize]";
}

/** Seconds of audio before the first provisional text. 0 would turn partials off. */
const INTERIM_SEC = "0.4";

// ---- rendering ---------------------------------------------------------------

const METER_WIDTH = 12;
const PARTIAL_WIDTH = 40;

/** Map RMS (~1e-4 quiet .. ~0.3 loud) to 0..1 on a log scale, which is how it sounds. */
function rmsFraction(rms: number): number {
	const v = Math.max(rms, 1e-4);
	return Math.min(1, Math.max(0, (Math.log10(v) + 4) / 3));
}

function quote(text: string, width = PARTIAL_WIDTH): string {
	return `"${text.length > width ? `${text.slice(0, width - 1)}…` : text}"`;
}

function renderMeter(rms: number, speech: boolean): string {
	const filled = Math.round(rmsFraction(rms) * METER_WIDTH);
	return `${speech ? "●" : "○"} ${"█".repeat(filled)}${"░".repeat(METER_WIDTH - filled)}`;
}

/**
 * Footer painter. `m` emits a level event every 30ms and each setStatus forces a full TUI
 * repaint, so identical bodies and rapid-fire level frames are dropped rather than drawn.
 */
const PAINT_INTERVAL_MS = 80;

class StatusPainter {
	private readonly ui: { setStatus(key: string, text: string | undefined): void } | null;
	private last = "";
	private at = 0;

	constructor(ctx: ExtensionCommandContext) {
		this.ui = ctx.hasUI ? ctx.ui : null;
	}

	set(body: string, throttle = false): void {
		if (!this.ui || body === this.last) return;
		const now = Date.now();
		if (throttle && now - this.at < PAINT_INTERVAL_MS) return;
		this.last = body;
		this.at = now;
		this.ui.setStatus("mouth", body);
	}

	clear(): void {
		if (!this.ui) return;
		this.last = "";
		this.at = 0;
		this.ui.setStatus("mouth", undefined);
	}
}

/** There is no UI outside the TUI (`pi -p`, rpc), and painting into one is a no-op at
 *  best. Every call that touches ctx.ui goes through hasUI first. */
function notify(
	ctx: ExtensionCommandContext,
	message: string,
	level: "info" | "warning" | "error",
): void {
	if (!ctx.hasUI) return;
	ctx.ui.notify(message, level);
}

// ---- the `m dictate --events` stream -------------------------------------------

interface MouthEvent {
	event?: string;
	rms?: number;
	speech?: boolean;
	text?: string;
	detail?: string;
	model?: string;
	message?: string;
	load?: number;
	threshold?: number;
}

/** Read newline-delimited JSON off a stream; anything else goes to onNoise. */
function onJsonLines(
	stream: NodeJS.ReadableStream,
	handle: (ev: MouthEvent) => void,
	onNoise: (line: string) => void,
): void {
	let buf = "";
	stream.setEncoding("utf8");
	stream.on("data", (chunk: string) => {
		buf += chunk;
		for (let nl; (nl = buf.indexOf("\n")) !== -1; ) {
			const line = buf.slice(0, nl).trim();
			buf = buf.slice(nl + 1);
			if (!line.startsWith("{")) {
				if (line) onNoise(line);
				continue;
			}
			try {
				handle(JSON.parse(line));
			} catch {
				onNoise(line);
			}
		}
	});
}

// ---- `/mouth install` -----------------------------------------------------------

/** Run a command, painting its last line of output into the footer as it goes. */
function run(
	bin: string,
	args: string[],
	painter: StatusPainter,
	label: string,
): Promise<{ code: number | null; tail: string }> {
	return new Promise((done) => {
		const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
		let tail = "";
		const absorb = (chunk: string) => {
			tail = `${tail}${chunk}`.slice(-4000);
			// uv reports progress on the last line; show that rather than a spinner that
			// says nothing during a multi-GB torch download.
			const line = chunk.trim().split("\n").pop()?.trim();
			if (line) painter.set(`◌ ${label} ${line.slice(0, 60)}`);
		};
		child.stdout!.setEncoding("utf8");
		child.stderr!.setEncoding("utf8");
		child.stdout!.on("data", absorb);
		child.stderr!.on("data", absorb);
		child.on("error", () => done({ code: null, tail }));
		child.on("close", (code) => done({ code, tail: tail.trim() }));
	});
}

async function install(ctx: ExtensionCommandContext): Promise<void> {
	const painter = new StatusPainter(ctx);
	const spec = `mouth${extrasForThisMachine()} @ ${REPO}`;
	painter.set("◌ installing mouth…");
	notify(ctx, `Installing ${spec}\nFirst run also downloads ~5GB of weights.`, "info");

	// --force so this doubles as the update path: uv resolves the git ref to a commit, and
	// without it an existing install is left alone and you are told nothing changed.
	const { code, tail } = await run(
		"uv",
		["tool", "install", "--force", spec],
		painter,
		"installing",
	);
	painter.clear();

	if (code === null) {
		notify(
			ctx,
			"Could not run `uv`. Install it first: https://docs.astral.sh/uv/getting-started/installation/",
			"error",
		);
		return;
	}
	if (code !== 0) {
		notify(ctx, `Install failed (${code}):\n${tail.slice(-600)}`, "error");
		return;
	}

	// uv puts the binary in its tool bin dir, which is not necessarily on this process's
	// PATH -- and a "success" that /mouth then cannot use is the worst of both.
	const probe = await run(M_BIN, ["--version"], painter, "checking");
	painter.clear();
	notify(
		ctx,
		probe.code === 0
			? `Installed ${probe.tail.trim()}. /mouth to start talking.`
			: `Installed, but \`${M_BIN}\` is not on pi's PATH yet.\nRun \`uv tool update-shell\` and restart pi, or set MOUTH_BIN.`,
		probe.code === 0 ? "info" : "warning",
	);
}

// ---- the command ---------------------------------------------------------------

export default function mouthExtension(pi: ExtensionAPI) {
	// A live listener is what "mouth mode is on" means; null is off.
	let listener: ChildProcess | null = null;
	let painter: StatusPainter | null = null;
	let ready = false; // has the mic actually opened?
	let base = ""; // committed transcript so far
	let partial = ""; // current interim
	let lastWritten = ""; // what WE last put in the editor -- anything else means the user sent or edited

	function paint(rms: number | null, speech: boolean): void {
		if (!listener || !painter) return;
		const committed = base ? `${quote(base.slice(-60), 60)} ` : "";
		const interim = partial ? quote(partial) : "listening… (↩ sends · /mouth stops)";
		const meter = rms == null ? "◌" : renderMeter(rms, speech);
		painter.set(`${meter} ${committed}${interim}`, rms != null);
	}

	function appendFinal(ctx: ExtensionCommandContext, text: string): void {
		// `m` traps SIGTERM as "I stopped talking" rather than "abort", so the utterance
		// in flight when you hit /mouth still gets transcribed and still emits a final on
		// a stderr we are still draining. Without this it lands in the editor seconds
		// after the footer said OFF.
		if (!listener || !ctx.hasUI) return;
		// Send/edit detection: if the editor no longer holds exactly what we wrote, the
		// user pressed Enter (pi cleared it) or typed something -- so their contents are
		// the new ground truth and our buffer restarts from there.
		const cur = ctx.ui.getEditorText().trimEnd();
		if (!lastWritten || cur !== lastWritten) base = cur;
		base = base ? `${base.trimEnd()} ${text}` : text;
		partial = "";
		ctx.ui.setEditorText(base);
		lastWritten = base;
	}

	function stop(): void {
		listener?.kill("SIGTERM");
		listener = null;
		painter?.clear();
		painter = null;
		ready = false;
	}

	function start(ctx: ExtensionCommandContext): void {
		base = "";
		partial = "";
		lastWritten = "";
		ready = false;
		painter = new StatusPainter(ctx);
		// Loading is 0.4s on a quantised MLX checkpoint but 6-10s on torch, and
		// calibration adds a second on a cold device. Say so rather than looking hung.
		painter.set("◌ starting listener…");

		let noise = "";
		const child = spawn(M_BIN, ["dictate", "--hold", "--events", "--interim", INTERIM_SEC], {
			stdio: ["ignore", "ignore", "pipe"],
		});
		listener = child;

		onJsonLines(
			child.stderr!,
			(ev) => {
				switch (ev.event) {
					case "loading":
					case "status":
						painter?.set(`◌ ${ev.detail ?? ev.model ?? "warming up…"}`);
						break;
					// The one that matters. Levels only start flowing once the mic is
					// open, so without this the footer sits on "starting listener…" for
					// the whole load and you talk into a device that isn't listening yet.
					case "ready":
						ready = true;
						notify(ctx, `🗣️ Listening — ready in ${(ev.load ?? 0).toFixed(1)}s.`, "info");
						paint(null, false);
						break;
					case "level":
						paint(ev.rms ?? 0, !!ev.speech);
						break;
					case "partial":
						partial = String(ev.text ?? "");
						paint(null, true);
						break;
					case "final":
						appendFinal(ctx, String(ev.text ?? ""));
						paint(null, false);
						break;
					case "error":
						notify(ctx, `mouth: ${ev.message}`, "warning");
						break;
				}
			},
			(line) => {
				// Keep a tail of anything that wasn't JSON: if the process dies, this is
				// the only explanation there will be.
				noise = `${noise}\n${line}`.slice(-300);
			},
		);

		child.on("error", (err) => {
			stop();
			notify(
				ctx,
				`Could not launch ${M_BIN}: ${err.message}\nRun /mouth install, or set MOUTH_BIN.`,
				"error",
			);
		});

		child.on("close", (code) => {
			// /mouth already cleared the listener when it tore this down, and a deliberate
			// stop does not need reporting.
			const deliberate = listener === null;
			const failedToStart = !ready;
			stop();
			if (deliberate) return;
			if (code === 0) {
				notify(ctx, "Mouth mode stopped.", "info");
				return;
			}
			const why = noise.trim() ? `\n${noise.trim()}` : "";
			notify(
				ctx,
				failedToStart
					? `Mouth mode never started (${M_BIN} exited ${code}).${why}`
					: `Mouth mode exited (${code}).${why}`,
				"error",
			);
		});

		notify(
			ctx,
			"Starting… every pause commits a sentence to the editor. ↩ Enter sends (the listener survives). /mouth to stop.",
			"info",
		);
	}

	pi.registerCommand("mouth", {
		description: "Toggle dictation — pauses commit sentences to the editor, ↩ sends",
		getArgumentCompletions: (prefix) =>
			"install".startsWith(prefix.trim())
				? [{ value: "install", label: "install", description: "Install the mouth CLI from its repo" }]
				: null,
		handler: async (args, ctx) => {
			const sub = args.trim();
			if (sub === "install") {
				await install(ctx);
				return;
			}
			if (sub) {
				notify(ctx, `Unknown: /mouth ${sub}. Use /mouth or /mouth install.`, "warning");
				return;
			}
			if (listener) {
				stop();
				notify(ctx, "Mouth mode OFF.", "info");
				return;
			}
			start(ctx);
		},
	});

	// The listener is a child process; leaving it holding the microphone after pi exits
	// is the one failure mode that outlives the session.
	pi.on("session_shutdown", async () => {
		stop();
	});
}
