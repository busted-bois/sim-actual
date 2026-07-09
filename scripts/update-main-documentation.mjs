#!/usr/bin/env node
/**
 * Rewrite docs/main-documentation.md via Cursor SDK (local agent).
 * Requires: CURSOR_API_KEY, prior `make doc-context`.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Agent, CursorAgentError } from "@cursor/sdk";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const DOC_PATH = join(ROOT, "docs", "main-documentation.md");
const PROMPT_PATH = join(ROOT, ".github", "doc-update-prompt.md");
const CONTEXT_PATH = join(ROOT, ".doc-context.txt");

function read(path) {
  if (!existsSync(path)) {
    console.error(`update-main-documentation: missing ${path}`);
    process.exit(1);
  }
  return readFileSync(path, "utf8");
}

function parseContextField(context, key) {
  const marker = `=== ${key} ===`;
  const start = context.indexOf(marker);
  if (start === -1) return "";
  const from = start + marker.length + 1;
  const next = context.indexOf("\n=== ", from);
  const chunk = next === -1 ? context.slice(from) : context.slice(from, next);
  return chunk.trim();
}

async function main() {
  const apiKey = process.env.CURSOR_API_KEY;
  if (!apiKey) {
    console.error("update-main-documentation: set CURSOR_API_KEY");
    process.exit(1);
  }

  const promptTemplate = read(PROMPT_PATH);
  const context = existsSync(CONTEXT_PATH) ? read(CONTEXT_PATH) : "";
  const currentDoc = read(DOC_PATH);
  const mainSha = parseContextField(context, "MAIN_SHA");
  const mainSubject = parseContextField(context, "MAIN_SUBJECT");

  const before = readFileSync(DOC_PATH, "utf8");

  const userPrompt = `${promptTemplate}

---

## Current documentation

\`\`\`markdown
${currentDoc}
\`\`\`

---

## Repository context

\`\`\`
${context}
\`\`\`

---

Update \`docs/main-documentation.md\` for main at \`${mainSha}\` — "${mainSubject}".
Edit the file directly in the workspace.`;

  console.log(`[doc-update] prompting agent for ${mainSha.slice(0, 7)}...`);

  let result;
  try {
    result = await Agent.prompt(userPrompt, {
      apiKey,
      model: { id: "composer-2.5" },
      local: { cwd: ROOT },
    });
  } catch (err) {
    if (err instanceof CursorAgentError) {
      console.error(`update-main-documentation: agent failed: ${err.message}`);
      process.exit(1);
    }
    throw err;
  }

  if (result.status === "error" || result.status === "cancelled") {
    console.error(`update-main-documentation: run ${result.status} (${result.id})`);
    process.exit(2);
  }

  let after = existsSync(DOC_PATH) ? readFileSync(DOC_PATH, "utf8") : "";

  // Fallback: agent returned full markdown in result text.
  if (after === before && result.result) {
    const fenced = result.result.match(/```markdown\n([\s\S]*?)```/);
    const body = fenced ? fenced[1] : result.result;
    if (body.includes("# Main Branch Documentation")) {
      writeFileSync(DOC_PATH, body.trimEnd() + "\n", "utf8");
      after = readFileSync(DOC_PATH, "utf8");
      console.log("[doc-update] wrote doc from agent result text");
    }
  }

  if (after === before) {
    console.error("update-main-documentation: doc unchanged after agent run");
    process.exit(3);
  }

  if (!after.includes("# Main Branch Documentation")) {
    console.error("update-main-documentation: output missing doc title");
    process.exit(4);
  }

  console.log(`[doc-update] done (${result.durationMs ?? "?"} ms)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
