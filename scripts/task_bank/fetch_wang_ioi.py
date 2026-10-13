"""Fetch the IOI generator of Wang et al. (redwoodresearch/Easy-Transformer) and the ACDC copy at pinned commits."""
from fetchlib import clone_pinned
clone_pinned('wang_ioi', {'Easy-Transformer': ('https://github.com/redwoodresearch/Easy-Transformer', 'ea15315dd24481e9e2ac5c3ef335d82907a1dc34'),
                          'Automatic-Circuit-Discovery': ('https://github.com/ArthurConmy/Automatic-Circuit-Discovery', 'bc99ace817974b5584b7ee203d596a8e2bbcd399')})
