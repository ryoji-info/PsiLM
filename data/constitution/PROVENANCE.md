# Provenance

- Source: https://www.anthropic.com/constitution (the `<article>` element of the page), fetched 2026-09-12 with curl.
- Licence: the page states that the constitution is released under the Creative Commons CC0 1.0 Deed (https://creativecommons.org/publicdomain/zero/1.0/) — free use for any purpose. The sentence is quoted in `page_preamble.md` / the article as fetched.
- `claudes_constitution.md`: the article body from the "Overview" heading to the end of "Acknowledgements", headings kept as Markdown (`##`/`###`/`####`), list items as `- `. Navigation, the "Read a summary" call-out and the site footer removed. 28958 words, sha256 dbcc88a041ae6f9daeda95e3b708b98598c895ff0fb91ce31902c0dad1c37052.
- `page_preamble.md`: the page's framing paragraphs above the article (not part of the constitution proper).
- Extraction script: the one-off Python snippet recorded in the session that produced this directory (regex over block-level tags, HTML entities unescaped, whitespace normalised); no wording was altered.
- Section count: 39 headings. Last headings: ['Concluding thoughts', 'Acknowledging open problems', 'On the word “constitution”', 'A final word'].
