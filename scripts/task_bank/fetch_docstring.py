"""Fetch the docstring prompt generator (acdc/docstring/prompts.py in the ACDC repo) at a pinned commit."""
from fetchlib import clone_pinned
clone_pinned('docstring', {'Automatic-Circuit-Discovery': ('https://github.com/ArthurConmy/Automatic-Circuit-Discovery', 'bc99ace817974b5584b7ee203d596a8e2bbcd399')})
