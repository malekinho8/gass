ga_settings ={
    'num_generations': 100,
    'num_parents_mating': 10,
    'sol_per_pop': 10,
    'crossover_type':'uniform',
    'mutation_type':'random',
    'mutation_percent_gene':10
}

note_to_midi = {
    'C0': 12, 'C#0': 13, 'Db0': 13, 'D0': 14, 'D#0': 15, 'Eb0': 15, 'E0': 16,
    'F0': 17, 'F#0': 18, 'Gb0': 18, 'G0': 19, 'G#0': 20, 'Ab0': 20, 'A0': 21,
    'A#0': 22, 'Bb0': 22, 'B0': 23,
    'C1': 24, 'C#1': 25, 'Db1': 25, 'D1': 26, 'D#1': 27, 'Eb1': 27, 'E1': 28,
    'F1': 29, 'F#1': 30, 'Gb1': 30, 'G1': 31, 'G#1': 32, 'Ab1': 32, 'A1': 33,
    'A#1': 34, 'Bb1': 34, 'B1': 35,
    'C2': 36, 'C#2': 37, 'Db2': 37, 'D2': 38, 'D#2': 39, 'Eb2': 39, 'E2': 40,
    'F2': 41, 'F#2': 42, 'Gb2': 42, 'G2': 43, 'G#2': 44, 'Ab2': 44, 'A2': 45,
    'A#2': 46, 'Bb2': 46, 'B2': 47,
    'C3': 48, 'C#3': 49, 'Db3': 49, 'D3': 50, 'D#3': 51, 'Eb3': 51, 'E3': 52,
    'F3': 53, 'F#3': 54, 'Gb3': 54, 'G3': 55, 'G#3': 56, 'Ab3': 56, 'A3': 57,
    'A#3': 58, 'Bb3': 58, 'B3': 59,
    'C4': 60, 'C#4': 61, 'Db4': 61, 'D4': 62, 'D#4': 63, 'Eb4': 63, 'E4': 64,
    'F4': 65, 'F#4': 66, 'Gb4': 66, 'G4': 67, 'G#4': 68, 'Ab4': 68, 'A4': 69,
    'A#4': 70, 'Bb4': 70, 'B4': 71,
    'C5': 72, 'C#5': 73, 'Db5': 73, 'D5': 74, 'D#5': 75, 'Eb5': 75, 'E5': 76,
    'F5': 77, 'F#5': 78, 'Gb5': 78, 'G5': 79, 'G#5': 80, 'Ab5': 80, 'A5': 81, 
    'A#5': 82, 'Bb5': 82, 'B5': 83,
    'C6': 84, 'C#6': 85, 'Db6': 85, 'D6': 86, 'D#6': 87, 'Eb6': 87, 'E6': 88,
    'F6': 89, 'F#6': 90, 'Gb6': 90, 'G6': 91, 'G#6': 92, 'Ab6': 92, 'A6': 93,
    'A#6': 94, 'Bb6': 94, 'B6': 95,
    'C7': 96, 'C#7': 97, 'Db7': 97, 'D7': 98, 'D#7': 99, 'Eb7': 99, 'E7': 100,
    'F7': 101, 'F#7': 102, 'Gb7': 102, 'G7': 103, 'G#7': 104, 'Ab7': 104, 'A7': 105,
    'A#7': 106, 'Bb7': 106, 'B7': 107,
    'C8': 108, 'C#8': 109, 'Db8': 109, 'D8': 110, 'D#8': 111, 'Eb8': 111, 'E8': 112,
    'F8': 113, 'F#8': 114, 'Gb8': 114, 'G8': 115, 'G#8': 116, 'Ab8': 116, 'A8': 117,
    'A#8': 118, 'Bb8': 118, 'B8': 119,
    'C9': 120, 'C#9': 121, 'Db9': 121, 'D9': 122, 'D#9': 123, 'Eb9': 123, 'E9': 124,
    'F9': 125, 'F#9': 126, 'Gb9': 126, 'G9': 127
}

tal_uno_categories = {
    "Synth Pads": ["PAD"],
    "Bass": ["BAS", "Bass"],
    "Leads": ["LED", "Lead"],
    "Arpeggios": ["ARP", "Arp"],
    "Pianos & Keyboards": ["PNO", "Piano", "KEY", "Keys", "KBD", "Celesta", "Clavinet", "Clavichord", "Harpsichord"],
    "Strings": ["STR", "Strings", "Violine"],
    "Organs": ["ORG", "Organ"],
    "Brass & Woodwinds": ["BRS", "Brass", "Horn", "Trumpet", "WND", "Flute", "Oboe", "Clarinet", "English Horn"],
    "Guitars": ["GTR", "Guitar"],
    "Miscellaneous/Other": ["SFX", "DRM", "PRC", "SYN", "MFX", "MT", "The Difference", "FN", "FMR"],
}

tal_uno_to_dawdreamer_mapping = {
    "@path": None,  # No suitable match
    "@programname": None,  # No suitable match
    "@category": None,  # No suitable match
    "@modulation": "modulation",
    "@dcolfovalue": "dco lfo value",
    "@dcopwmvalue": "dco pwm value",
    "@dcopwmmode": "dco pwm mode",
    "@dcopulseenabled": "dco pulse enabled",
    "@dcosawenabled": "dco saw enabled",
    "@dcosuboscenabled": "dco sub osc enabled",
    "@dcosuboscvolume": "dco sub osc volume",
    "@dconoisevolume": "dco noise volume",
    "@hpfvalue": "dco hp filter",
    "@filtercutoff": "filter cutoff",
    "@filterresonance": "filter resonance",
    "@filterenvelopemode": "filter env mode",
    "@filterenvelopevalue": "filter env",
    "@filtermodulationvalue": "filter modulation",
    "@filterkeyboardvalue": "filter keyboard",
    "@volume": "master volume",
    "@masterfinetune": "master fine tune",
    "@octavetranspose": "master octave transpose",
    "@vcamode": "vca mode",
    "@adsrattack": "attack",
    "@adsrdecay": "decay",
    "@adsrsustain": "sustain",
    "@adsrrelease": "release",
    "@lforate": "lfo rate",
    "@lfodelaytime": "lfo delay",
    "@lfotriggermode": "lfo trigger mode",
    "@lfomanualtriggerenabled": "lfo trigger enabled",
    "@lfomanualtriggeractive": "lfo trigger active",
    "@lfowaveform": "lfo waveform",
    "@chorus1enable": "chorus 1",
    "@chorus2enable": "chorus 2",
    "@arpenabled": "arp enabled",
    "@arpsyncenabled": "arp sync enabled",
    "@arpmode": "arp mode",
    "@arprange": "arp range",
    "@arprate": "arp rate",
    "@arpnotloadsettings": "arp locked",
    "@controlvelocityvolume": "control velocity volume",
    "@controlvelocityenvelope": "control velocity envelope",
    "@controlbenderfilter": "control pitch bend filter",
    "@controlbenderdco": "control pitch bend dco",
    "@portamentomode": "portamento mode",
    "@portamentointensity": "portamento intensity",
    "@midilearn": "midi learn",
    "@panic": "panic",
    "@voicehold": "voice hold",
    "@miditriggerarp16sync": "trigger arp by midi channel 16",
    "@midiclocksync": "clock sync",
    "@hostsync": "host sync",
    "@maxpoly": "max voices",
    "@keytranspose": "keytranspose",
    "@arpsyncmode": None,  # No suitable match
    "@arpspecialmode": "special mode",
    "@lfoinverted": "lfo inverted",
    "@portamentopoly": "portamento poly",
    "@engineoff": "sound engine off",
    "@pitchwheel": "pitch wheel",
    "@modulationwheel": "modulation wheel",
    "@midiclear": "midi clear",
    "@midilock": "midi lock",
    "@mpeEnabled": "MPE enabled",
    "@portamentotimeenabled": "portamento time",
    "@reverbDryWet": "FX Reverb Dry / Wet",
    "@reverbSize": "FX Reverb Size",
    "@reverbDelay": "FX Reverb Delay",
    "@reverbTone": "FX Reverb Tone",
    "@delayDryWet": "FX Delay Dry / Wet",
    "@delayTime": "FX Delay Time",
    "@delaySync": "FX Delay Sync",
    "@delaySpread": "FX Delay Spread",
    "@delayTone": "FX Delay Tone",
    "@delayFeedback": "FX Delay Feedback",
    "@mtsEnabled": "MTS Microtuning Active",
    "@unisonovoices": "Unisono Voices",
    "@unisonodetune": "Unisono Detune",
    "@unsionospread": "Unisono Spread",
    "@voicemode": "Voice Mode",
    "tuningtable": None,  # No suitable match
    "voicetunings": None,  # No suitable match
}

tal_uno_to_dawdreamer_index_mapping = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:9,
    10:10,
    11:11,
    12:12,
    13:13,
    14:14,
    15:15,
    16:16,
    17:17,
    18:18,
    19:19,
    20:20,
    21:21,
    22:22,
    23:23,
    24:24,
    25:25,
    26:26,
    27:27,
    28:28,
    29:29,
    30:30,
    31:31,
    32:32,
    33:33,
    34:34,
    35:35,
    36:36,
    37:37,
    38:38,
    39:39,
    40:40,
    41:41,
    42:42,
    43:43,
    44:44,
    45:45,
    46:46,
    47:47,
    48:48,
    49:49,
    50:50,
    51:51,
    52:53,
    53:54,
    54:55,
    55:56,
    56:59,
    57:60,
    58:61,
    59:62,
    60:63,
    61:64,
    62:66,
    63:67,
    64:68,
    65:69,
    66:70,
    67:71,
    68:72,
    69:73,
    70:74,
    71:75,
    72:76,
    73:77,
    74:78,
    75:79,
    76:80,
} 