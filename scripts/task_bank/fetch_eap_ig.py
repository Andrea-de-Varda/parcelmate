"""Fetch the EAP-IG task set: hannamw/eap-ig-faithfulness (data/) and hannamw/EAP-IG (greater_than_data.csv, ioi_llama.csv) at pinned commits."""
from fetchlib import clone_pinned
clone_pinned('eap_ig', {'eap-ig-faithfulness': ('https://github.com/hannamw/eap-ig-faithfulness', 'd42fea6aa2c7acca39007c9c94923770669cb9ae'),
                        'EAP-IG': ('https://github.com/hannamw/EAP-IG', 'e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd'),
                        'gpt2-greater-than': ('https://github.com/hannamw/gpt2-greater-than', '0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a')})
