#!/bin/sh
#
# Compute a darktable-style version string from the repo's git history.
# Mirrors darktable/tools/get_git_version_string.sh so model releases
# carry the same version shape as the host app.
#
# Examples of output:
#   release-5.6.0 tagged commit      -> "5.6.0"
#   47 commits past release-5.6.0    -> "5.6.0+47~gXXXXXXX"
#   dirty working tree               -> "5.6.0+47~gXXXXXXX~dirty"
#   no release-* tag at all          -> bare commit hash from git describe
#   not a git repo / unknown         -> "unknown-version" (exits 0)
#
# Used by CI workflows for the nightly version label and by maintainer
# scripts that need to print "what version am I on".

# Use semver-sorted tag list; `git describe` breaks ties between tags
# on the same commit by tagger date, which picks the wrong tag when
# 5.7.0 was anchored before 5.6.0 was released.

LATEST_TAG="$(git tag --sort=-version:refname --merged HEAD --list 'release-*' 2>/dev/null | head -n 1)"

if [ -n "$LATEST_TAG" ] ;
then
  VERSION="${LATEST_TAG#release-}"
  COMMITS_SINCE="$(git rev-list "$LATEST_TAG"..HEAD --count 2>/dev/null)"
  COMMITS_SINCE="${COMMITS_SINCE:-0}"   # treat git failure as "on the tag"
  SHORT_HASH="$(git rev-parse --short HEAD 2>/dev/null)"
  SHORT_HASH="${SHORT_HASH:-unknown}"

  DIRTY=""
  if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null ; then
    DIRTY="~dirty"
  fi

  if [ "$COMMITS_SINCE" = "0" ] && [ -z "$DIRTY" ] ; then
    echo "$VERSION"
  elif [ "$COMMITS_SINCE" = "0" ] ; then
    echo "${VERSION}${DIRTY}"
  else
    echo "${VERSION}+${COMMITS_SINCE}~g${SHORT_HASH}${DIRTY}"
  fi
  exit 0
fi

# shallow clones may have no tags; fall back to the bare commit hash
VERSION="$(git describe --always --dirty 2>/dev/null)"
if [ $? -eq 0 ] ;
then
  echo "$VERSION"
  exit 0
fi

echo "unknown-version"
exit 0
