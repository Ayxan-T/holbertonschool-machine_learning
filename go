#!/usr/bin/env bash

chmod u+x "$1"

git add "$1"
git commit -m "autocommit"
git push
