#!/usr/bin/env bash


FILE=~/.jupyter/jupyter_notebook_config.py
if [ ! -f "$FILE" ]; then
    echo "jupyter_notebook_config.py not found. generating"
    jupyter notebook --generate-config
fi

read -r -d '' VAR << EOM
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)

c.FileContentsManager.post_save_hook = post_save
EOM

echo "$VAR" | cat - $FILE > temp && mv temp $FILE