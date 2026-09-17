import { createHash } from "node:crypto";
import { readdir, readFile, writeFile } from "node:fs/promises";

// Stamp the whole module graph and stylesheet together. Run after any dist edit.
const dist = new URL("../dist/", import.meta.url);
const names = (await readdir(dist)).filter(name => /\.(?:js|css|html)$/.test(name)).sort();
const sources = new Map(await Promise.all(names.map(async name => [name,
  (await readFile(new URL(name, dist), "utf8")).replace(/(\.(?:js|css))\?v=[a-f0-9]+/g, "$1")
])));
const hash = createHash("sha256");
for (const [name, source] of sources) hash.update(name + "\0" + source);
const version = hash.digest("hex").slice(0, 12);
for (const [name, source] of sources) {
  const stamped = name.endsWith(".js")
    ? source.replace(/((?:from\s*|import\s*)["'])(\.\/[^"']+\.js)(["'])/g, `$1$2?v=${version}$3`)
    : name.endsWith(".html")
      ? source.replace(/((?:src|href)=["'])(\/[^"']+\.(?:js|css))(["'])/g, `$1$2?v=${version}$3`)
      : source;
  await writeFile(new URL(name, dist), stamped);
}
console.log(`Dashboard release ${version}`);
