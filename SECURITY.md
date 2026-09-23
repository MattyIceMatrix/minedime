# Security

The site is static: one HTML file served by GitHub Pages. It has no server, stores nothing, holds no credentials,
and makes no network requests. Fonts are embedded.

A Content-Security-Policy allows only the page's own two scripts, pinned by SHA-256 hash, and blocks every other
script, frame, form, and outbound connection. Injected scripts do not run.

To report a problem, open an issue without sensitive details and a private channel will be arranged.
