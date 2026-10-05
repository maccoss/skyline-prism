# Publishing to Panorama

PRISM can publish a finished output directory to [Panorama](https://panoramaweb.org) in one step:

- the **QC report** as a wiki page, in a Panorama folder you choose;
- the **quant report** as a wiki page, in a folder you choose (it may be a different one);
- the **whole output directory**, uploaded next to the folder that holds the experiment's raw files;
- a **links page**, shown on a folder's own page, that links the QC page, every quant page
  published from the output directory, and the uploaded files.

The Skyline tool's **Publish to Panorama...** button (bottom right, beside Open QC Report) and the
`prism publish` command do the same thing through the same code (`Core/Panorama/OutputPublishing`).
The window's **Show Command Line** gives the command it would run.

## The command line

```bash
prism publish -d output_dir/ \
    --qc-wiki /MacCoss/maccoss/My-Project \
    --quant-wiki /MacCoss/maccoss/My-Project/Analysis \
    --beside-raw /MacCoss/maccoss/My-Project/@files/RawFiles
```

| Option | What it does |
|---|---|
| `--qc-wiki FOLDER` | Publishes `qc_report.html` as a wiki page in that folder |
| `--quant-wiki FOLDER` | Publishes `quant/quant_report.html` as a wiki page in that folder |
| `--beside-raw FOLDER` | Uploads the output directory next to this raw files folder, under its own name |
| `--links-wiki FOLDER` | Publishes the links page in that folder and shows it on the folder's page (default: the QC page's folder) |
| `--qc-page NAME`, `--quant-page NAME`, `--links-page NAME` | Page names (defaults below) |
| `--server URL` | Another Panorama server (default `https://panoramaweb.org`) |
| `--replace-edited` | Replace a page that was edited on Panorama, a same-named page PRISM did not write, or one published from a different output directory |
| `--no-upload`, `--no-qc`, `--no-quant`, `--no-links` | Skip a step remembered from an earlier publish (or, for the links page, the default), this once |
| `--dry-run` | Print what would happen and send nothing |

A FOLDER can be a path, or an address pasted from a browser
(`https://panoramaweb.org/MacCoss/maccoss/My-Project/project-begin.view`). Every publish records its
targets in the output directory's `panorama.json`, so `prism publish -d output_dir/` on its own
republishes to the same places. Show Command Line writes a step the window skips as its `--no-` flag,
because the command line would otherwise fill it in from `panorama.json`.

## Where things go

**Wiki pages.** Each page goes in the wiki of the folder you name. A file-area path is reduced to its
folder. The default names are `PRISM-QC-<output directory>` and
`PRISM-Quant-<output directory>-<contrast>`. The quant page carries the contrast in its name because
one output directory is often analyzed with several contrasts, and the quant report in it is replaced
by each new one. A generated quant page name follows the current contrast, so publishing a new
contrast never overwrites the previous contrast's page. A name you typed yourself is kept as given.

**The output directory.** It goes beside the raw files folder, not inside it:
`.../@files/RawFiles` gives `.../@files/<output directory name>`. If the raw files sit directly in the
file root (`.../@files`), the output directory goes inside that root, since a root has no sibling.
Subfolders keep their layout. The upload happens first, so both wiki pages can link to it. The quant
page links to the uploaded `quant/` folder, where its tables are.

**The links page, on the folder's page.** The report pages are ordinary wiki pages, so nothing on a
folder's own page leads to them. PRISM therefore also publishes a small page, `PRISM-<output
directory>` by default, in the QC page's folder unless told otherwise. It lists the QC page, every
quant page published from this output directory that is still on Panorama (one per contrast), and the
uploaded output directory. PRISM then shows that page in a **Wiki web part on the folder's page**, placed right after the
**Targeted MS Runs** part, so the results sit with the documents they came from. In a Panorama folder
that puts it between Targeted MS Runs and **Files**. A folder without a Targeted MS Runs part gets it
right above Files, and one with neither gets it at the top. The web part's id is recorded in
`panorama.json`. A republish updates the same part, and leaves it wherever someone has moved it since.
A part someone removed is added again, in the same place as a new one.

Adding a web part needs **folder administrator** permission. Without it, the links page is still
published along with everything else, and the publish says what is missing: a folder administrator
can add a Wiki web part showing that page by hand. The links step never fails the publish: if the
links page itself cannot be written, the reports and the upload stand, and the publish says why.

## Republishing

- **Pages are updated in place.** Panorama keeps every version of a page's text. Its plots are page
  attachments, and Panorama does not version those: they belong to the page, not to a version of it.
  So each plot is attached under a name that includes a hash of its image. A plot that did not change
  keeps its name and is not sent again. A changed one goes up under a new name, and the one it
  replaced is removed only after the new page is saved. An earlier version opened from the page's
  history therefore shows the plots it shares with the current one, and a missing image for any that
  changed. It never shows another run's plot in their place.
- **A page edited on Panorama is not overwritten.** PRISM marks each page it writes with a footer that
  carries a fingerprint of what it wrote. If someone has since edited the page on Panorama, or a page
  of that name exists that PRISM never wrote, the publish stops and says so.
- **A page belongs to the output directory that published it.** Default page names come from the
  output directory's name, and names like `PRISM-sum-rtlowess-medianpolish` recur from one experiment
  to the next. So each output directory gets a random id on its first publish (`publish_id` in
  `panorama.json`), and every page it publishes carries that id in its footer. Another output
  directory with the same name, publishing to the same folder, is refused rather than allowed to
  replace the first one's page: choose another page name for it.
- **Replacing one on purpose.** In any of these cases, pass `--replace-edited` (or tick **Replace a
  page edited on Panorama**). The text it replaces stays in the page's history.
- **Only changed files are uploaded.** Before uploading, each file is compared with Panorama's own MD5
  of its copy (LabKey's `?method=md5sum`, computed over the bytes the server stored). A file already
  there unchanged is skipped. After a folder's files go up, they are checked the same way. A file that
  did not arrive intact is sent once more, and then the publish is refused. A dropped connection, or
  a 502, 503 or 504 from the gateway in front of Panorama, is retried up to three times, from the
  start of the file.
- **Nothing is deleted on Panorama.** A file you removed from the local output directory stays in the
  uploaded copy until you delete it there.

## Signing in

PRISM tries, in order:

1. an API key in the environment variable **`PRISM_PANORAMA_API_KEY`**: for headless runs, and the
   only option on Linux and macOS;
2. the sign-in **PanoramaBridge** saved on this computer (Windows Credential Manager);
3. the one **LabOps** saved;
4. the one the **Skyline tool** saved.

On a lab computer that already runs PanoramaBridge, nothing needs to be set up. Otherwise the Publish
window asks once for an API key, or a Panorama email and password, and saves it in Windows Credential
Manager as `Skyline-PRISM:https://panoramaweb.org`. It never touches PanoramaBridge's or LabOps's
entries. An API key is generated on Panorama from your name's menu, under **External Tool Access**.
`panorama.json` holds folder and page names only, never a credential, because it is uploaded with
the outputs.

## What Panorama requires, and what PRISM does about it

These were measured against panoramaweb.org (October 2026) rather than taken from the WebDAV or LabKey
documentation, which differ from it in ways that matter.

- **No `<style>` block, `<link>`, `<script>` or form element in a wiki page**, no `on*` attribute, and
  no `url()` in a style attribute. LabKey refuses the whole page from anyone who is not a trusted
  developer on the server (`PageFlowUtil.validateHtml`). PRISM's reports keep their CSS in a
  `<style>` block, so publishing applies it to each element as a `style` attribute, using a real CSS
  selector engine (AngleSharp) and specificity order, and removes the block. The report's class names
  are dropped too. Panorama's own stylesheet (Bootstrap) also styles `.container`, `.box` and
  `.note`, and would otherwise rearrange the page.
- **The plots become page attachments**, not embedded data. They are most of a report's size (27 plots,
  about 2 MB, in a 96-sample QC report), and Panorama keeps every page version, so embedded plots
  would be stored again with every republish.
- **An uploaded `.html` file faces the same rules, and a `<!doctype>` is refused as well.** The same
  bytes named `.txt` are accepted, and of doctype, `<html>`, `<head>`, `<meta>` and `<title>`, only
  the doctype is refused. So `qc_report.html` and `quant_report.html` are uploaded in a form Panorama
  accepts: the same page with its styles inlined, its images still embedded so the file stands
  alone, and no doctype. Every other file is uploaded byte for byte.
- **A wiki save needs LabKey's CSRF token, even with an API key.** PRISM gets one from
  `login-whoami.api` for each save, together with that session's cookie.
- **Attaching a file whose name is already taken is refused with a warning, not replaced.** PRISM
  removes the previous publish's attachments first, in a request of its own.
- **Creating a folder that already exists answers 200** on panoramaweb.org, where PanoramaBridge
  measured 405 earlier. Both are treated as success.
- **A semicolon in a file name** makes the server truncate the name, so such a file is refused before
  anything is sent.
- **A Panorama folder's page is not `portal.default`.** A Targeted MS folder keeps its page's web parts
  under the page id `DefaultDashboard`, and `portal.default` lists nothing there. PRISM uses the page
  whose parts the folder's start page actually renders.
- **Adding a web part takes the column by LabKey's internal name.** `project-addWebPart.view` is a form
  action, not an API. It answers with a redirect and never names the new part. It wants the body
  column as `!content`, though the web part listing reports that column as `body`. Sent `body`, it
  saves the part anyway, in a column no page renders and the listing leaves out, so the request
  answers exactly as a success does and nothing appears. PRISM sends the internal name and finds the
  new part by listing the page before and after. The part's settings (which page it shows) go through
  `project-customizeWebPartAsync.api`.
