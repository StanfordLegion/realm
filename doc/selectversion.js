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
  const response = await fetch('https://api.github.com/repos/stanfordlegion/realm/contents/doc?ref=gh-pages');
  const data = await response.json();
  entries=[thisvers];
  if (thisvers != master) {
    entries.push(master);
  }
  data.forEach(function(entry) {
      if (entry.name.startsWith('v') && entry.name != thisvers) {
        entries.push(entry.name);
      }
  });
  entries = entries.map((x) => '<a class="verslink" href="'
                    + patharr.slice(0, urlrootdirs).join('/')
                    + '/' + x + '/">'
                    + url2label(x)
                    + '</a>');
  $('.dropdown-content').append(entries.join(''));
})();
