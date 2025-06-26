/// (c) Axel Naumann, CERN; 2020-03-02
/// Copyright 2024 Stanford University, NVIDIA Corporation
/// SPDX-License-Identifier: LGPL-3.0-only

/// Configurable section.

// What the master is called. Leave untouched if master has no doc.
let master = 'main';

/// Pathname part of the URL containing the different versioned doxygen
/// subdirectories. Must be browsable.
let urlroot = 'realm/doc';

// Convert a url directory (e.g. "v620") to a version number displayed on the
// web page (e.g. "6.20").
function url2label(versdir) {
   return versdir;
}

///=============================================================================
// Remove trailing '/'.
if (urlroot[urlroot.length - 1] == '/')
urlroot = urlroot.substring(0, urlroot.length - 1)
let urlrootdirs = urlroot.split('/').length;

function url2version(patharr) {
   // Given the directory array of a URL (i.e. without domain), extract the
   // version corresponding to the URL.
   // E.g. for `https://example.com/doc/master/classX.html`, the directory array
   // becomes `["doc", "master, "classX.html"]. This function might return
   // the second element, `master`.
   return patharr[patharr.length - urlrootdirs];
}

let patharr = window.location.pathname.replace(/\/+/g, '/').split('/');
let thisvers = url2version(patharr);
$('.dropbtn').html("Version " + url2label(thisvers));

// https://stackoverflow.com/questions/39048654
(async () => {
  const response = await fetch('/realm/doc/doc-versions')
    .then((response) => response.text())
    .then((data) => {
      entries = data.split('\n');
      entries = entries.splice(entries.indexOf(thisvers), -1);
      if (thisvers != master) {
        entries = entries.splice(entries.indexOf(master), -1);
        entries.unshift(master);
      }
      entries.unshift(thisvers);
      entries = entries.map((x) => '<a class="verslink" href="'
                        + '/' + urlroot + '/' + x + '/">'
                        + url2label(x)
                        + '</a>');
      $('.dropdown-content').append(entries.join(''));
    });
})();
