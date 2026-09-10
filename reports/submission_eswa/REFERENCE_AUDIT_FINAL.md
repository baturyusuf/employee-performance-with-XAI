# Final reference audit

## Archana et al. record

The article PDF is the controlling source for the ambiguous author metadata. Its first page prints the authors as **Archana Boob, Sandeep Sharma, Saurabh Singh, and Rafsan Ali**, with 2019, volume 7, Special Issue 14, pages 443–447.

- [Primary article PDF](https://www.ijcseonline.org/spl_pub_paper/93-IACIT%20-%20348.pdf)
- [Publisher article record](https://ijcseonline.org/index.php/j/article/view/5706)

The publisher platform's migrated structured metadata reverses some names and reports a later migration/publication date. The article PDF and issue evidence support the 2019 printed record. The DOI string printed in the PDF does not resolve through the DOI registry and is therefore omitted rather than asserted.

The repaired BibTeX author field is:

```bibtex
author = {Boob, Archana and Sharma, Sandeep and Singh, Saurabh and Ali, Rafsan}
```

Final rendered checks must reject `B. et al., 2019`, `B., A., S., S.`, and similar initial-only parsing failures. All other reference checks from `qc/REFERENCE_AUDIT.md` remain in force and will be rerun after document generation.

