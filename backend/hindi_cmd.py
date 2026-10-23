"""
Hindi and Hinglish command understanding.

JARVIS's tools (timers, weather, news, memory, tasks, the data engine ...) understand English. When you speak Hindi,
Speech recognition writes Devanagari ("पाँच मिनट का टाइमर लगाओ", "सेल्स का फोरकास्ट दिखाओ"), and when you type Hinglish you
write "5 minute ka timer lagao". normalize() turns those into the English command the tools already know, and leaves
everything else exactly as it was, so ordinary Hindi conversation still goes to the language model untouched and
plain English commands are never rewritten.

    normalize("पाँच मिनट का टाइमर लगाओ")          -> "set a 5 minutes timer"
    normalize("10 minute baad chai yaad dilana")   -> "remind me in 10 minutes to chai"
    normalize("दिल्ली का मौसम कैसा है")              -> "what is the weather in Delhi"
    normalize("आज की खबरें सुनाओ")                  -> "latest news"
    normalize("सेल्स का फोरकास्ट दिखाओ")            -> "sales का forecast दिखाओ"  (loanwords back to English)
"""

from __future__ import annotations

import re

_DEVANAGARI = "ऀ-ॿ"
_DEV = re.compile(f"[{_DEVANAGARI}]")
_W = f"\\w{_DEVANAGARI}"                      # a word character in Hindi or English (\w alone misses vowel signs)
_NOT_BEFORE = f"(?<![{_W}])"
_NOT_AFTER = f"(?![{_W}])"

# ------------------------------------------------------------------ numbers

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_HI_NUMBERS = """
0 शून्य | 1 एक | 2 दो | 3 तीन | 4 चार | 5 पाँच पांच | 6 छह छः छे छ | 7 सात | 8 आठ | 9 नौ | 10 दस
11 ग्यारह | 12 बारह | 13 तेरह | 14 चौदह | 15 पंद्रह पन्द्रह | 16 सोलह | 17 सत्रह | 18 अठारह | 19 उन्नीस | 20 बीस
21 इक्कीस | 22 बाईस बाइस | 23 तेईस तेइस | 24 चौबीस | 25 पच्चीस पचीस | 26 छब्बीस | 27 सत्ताईस सताईस
28 अट्ठाईस अठाईस अट्ठाइस | 29 उनतीस | 30 तीस | 31 इकतीस | 32 बत्तीस | 33 तैंतीस | 34 चौंतीस | 35 पैंतीस
36 छत्तीस | 37 सैंतीस | 38 अड़तीस | 39 उनतालीस उन्तालीस | 40 चालीस | 41 इकतालीस | 42 बयालीस | 43 तैंतालीस
44 चवालीस चौवालीस | 45 पैंतालीस | 46 छियालीस | 47 सैंतालीस | 48 अड़तालीस | 49 उनचास | 50 पचास | 51 इक्यावन
52 बावन | 53 तिरपन तिरेपन | 54 चौवन | 55 पचपन | 56 छप्पन | 57 सत्तावन | 58 अट्ठावन अठावन | 59 उनसठ | 60 साठ
61 इकसठ | 62 बासठ | 63 तिरसठ | 64 चौंसठ | 65 पैंसठ | 66 छियासठ | 67 सड़सठ | 68 अड़सठ | 69 उनहत्तर | 70 सत्तर
71 इकहत्तर | 72 बहत्तर | 73 तिहत्तर | 74 चौहत्तर | 75 पचहत्तर | 76 छिहत्तर | 77 सतहत्तर | 78 अठहत्तर | 79 उन्यासी
80 अस्सी | 81 इक्यासी | 82 बयासी | 83 तिरासी | 84 चौरासी | 85 पचासी | 86 छियासी | 87 सत्तासी | 88 अट्ठासी
89 नवासी | 90 नब्बे | 91 इक्यानवे | 92 बानवे | 93 तिरानवे | 94 चौरानवे | 95 पचानवे | 96 छियानवे | 97 सत्तानवे
98 अट्ठानवे | 99 निन्यानवे | 100 सौ
"""
_ROMAN_NUMBERS = """
1 ek | 2 do | 3 teen tin | 4 char chaar | 5 paanch panch paach | 6 chhe chhah che chah | 7 saat sat | 8 aath ath | 9 nau
10 das | 11 gyarah gyaarah | 12 barah baarah | 13 terah | 14 chaudah | 15 pandrah pandra | 16 solah | 17 satrah
18 atharah | 19 unnis | 20 bees | 25 pachchis pachis | 30 tees | 35 paintees | 40 chalis | 45 paintalis | 50 pachaas pachas
55 pachpan | 60 saath | 90 nabbe | 100 sau
"""


def _build_numbers() -> dict:
    out = {}
    for block in (_HI_NUMBERS, _ROMAN_NUMBERS):
        for chunk in block.replace("\n", "|").split("|"):
            parts = chunk.split()
            if len(parts) >= 2 and parts[0].isdigit():
                for w in parts[1:]:
                    out[w] = int(parts[0])
    return out


_NUM_WORDS = _build_numbers()

_UNIT = {
    "मिनट": "minutes", "मिनिट": "minutes", "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "minit": "minutes", "घंटे": "hours", "घंटा": "hours", "घण्टे": "hours", "घण्टा": "hours", "ghante": "hours",
    "ghanta": "hours", "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "सेकंड": "seconds",
    "सेकेंड": "seconds", "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds",
}
_WORD_ALT = "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True))
_UNIT_ALT = "|".join(sorted(map(re.escape, _UNIT), key=len, reverse=True))
# "दो" / "do" also mean "give" (laga do, kar do), so a number word counts only when a unit follows: "दो मिनट", "teen ghante"
_NUM_RE = re.compile(f"{_NOT_BEFORE}({_WORD_ALT})(?=\\s*(?:{_UNIT_ALT}){_NOT_AFTER})", re.I)
_NUM_LOOSE_RE = re.compile(f"{_NOT_BEFORE}({_WORD_ALT}){_NOT_AFTER}", re.I)
_AMBIGUOUS = {"do", "दो", "sat", "das", "tin", "ath", "che", "chah"}

# fractions of an hour that are said as one word
_HOUR = "(?:घंटे|घंटा|घण्टे|घण्टा|ghante|ghanta|hours?)"
_FRACTIONS = [
    (f"(?:डेढ़|डेढ|derh|dedh)\\s*{_HOUR}", "1.5 hours"),
    (f"(?:ढाई|dhai|adhai)\\s*{_HOUR}", "2.5 hours"),
    (f"(?:सवा|sawa|sava)\\s*{_HOUR}", "1.25 hours"),
    (f"(?:पौन|paun|pauna)\\s*{_HOUR}", "45 minutes"),
    (f"(?:आधा|आधे|aadha|aadhe|adha)\\s*{_HOUR}", "30 minutes"),
    ("(?:आधा|आधे|aadha|aadhe|adha)\\s*(?:मिनट|minute)", "30 seconds"),
]


def numberize(text: str, loose: bool = False) -> str:
    """Devanagari digits and Hindi number words -> digits: 'पाँच मिनट' -> '5 मिनट'.
    loose=True also converts number words that are not followed by a unit (for sums), except the ambiguous ones."""
    t = (text or "").translate(_DEV_DIGITS)
    for pattern, repl in _FRACTIONS:
        t = re.sub(pattern, repl, t, flags=re.I)

    def swap(m):
        w = m.group(1)
        return str(_NUM_WORDS.get(w, _NUM_WORDS.get(w.lower())))

    t = _NUM_RE.sub(swap, t)
    if loose:
        t = _NUM_LOOSE_RE.sub(lambda m: m.group(0) if m.group(1).lower() in _AMBIGUOUS else swap(m), t)
    return t


# ------------------------------------------------------------------ loanwords (Devanagari -> English)

_LOAN = {
    "डेटा": "data", "डाटा": "data", "डेटासेट": "dataset", "डाटासेट": "dataset", "डेटाबेस": "database", "फाइल": "file",
    "फ़ाइल": "file", "चार्ट": "chart", "चार्ट्स": "charts", "ग्राफ": "graph", "ग्राफ़": "graph", "प्लॉट": "plot",
    "डैशबोर्ड": "dashboard", "डेशबोर्ड": "dashboard", "रिपोर्ट": "report", "एक्सपोर्ट": "export", "डाउनलोड": "download",
    "अपलोड": "upload", "एनालिसिस": "analysis", "एनालाइज": "analyze", "एनालाइज़": "analyze", "विश्लेषण": "analysis",
    "फोरकास्ट": "forecast", "फ़ोरकास्ट": "forecast", "प्रेडिक्ट": "predict", "प्रेडिक्शन": "prediction",
    "भविष्यवाणी": "prediction", "पूर्वानुमान": "forecast", "मॉडल": "model", "ट्रेन": "train", "ट्रेनिंग": "training",
    "क्लस्टर": "cluster", "क्लस्टरिंग": "clustering", "आउटलायर": "outlier", "आउटलाइयर": "outlier", "एनोमली": "anomaly",
    "कोरिलेशन": "correlation", "सहसंबंध": "correlation", "रिग्रेशन": "regression", "क्लासिफिकेशन": "classification",
    "क्लासिफाई": "classify", "एक्यूरेसी": "accuracy", "फीचर": "feature", "फ़ीचर": "feature", "इम्पोर्टेंस": "importance",
    "टारगेट": "target", "वैरिएबल": "variable", "कॉलम": "column", "कॉलम्स": "columns", "रो": "row", "रोज़": "rows",
    "रोज": "rows", "सेल्स": "sales", "रेवेन्यू": "revenue", "प्रॉफिट": "profit", "प्रोफिट": "profit", "कस्टमर": "customer",
    "कस्टमर्स": "customers", "प्रोडक्ट": "product", "प्रोडक्ट्स": "products", "रीजन": "region", "रीज़न": "region",
    "ट्रेंड": "trend", "ग्रोथ": "growth", "मार्जिन": "margin", "डिस्काउंट": "discount", "ऑर्डर": "order",
    "ऑर्डर्स": "orders", "एवरेज": "average", "टोटल": "total", "मीन": "mean", "मिसिंग": "missing", "क्लीन": "clean",
    "वैल्यू": "value", "वैल्यूज़": "values", "इनसाइट": "insight", "इनसाइट्स": "insights", "सेगमेंट": "segment",
    "सेगमेंट्स": "segments", "हिस्टोग्राम": "histogram", "स्कैटर": "scatter", "हीटमैप": "heatmap", "टाइमर": "timer",
    "अलार्म": "alarm", "रिमाइंडर": "reminder", "बैटरी": "battery", "वॉल्यूम": "volume", "म्यूट": "mute",
    "वेदर": "weather", "टेम्परेचर": "temperature", "न्यूज़": "news", "न्यूज": "news", "हेडलाइन": "headlines",
    "हेडलाइंस": "headlines", "सर्च": "search", "गूगल": "google", "यूट्यूब": "youtube", "व्हाट्सएप": "whatsapp",
    "इंस्टाग्राम": "instagram", "डॉक्यूमेंट": "document", "डॉक्युमेंट": "document", "पीडीएफ": "pdf", "पीडीएफ़": "pdf",
    "सारांश": "summary", "समरी": "summary", "स्क्रीन": "screen", "स्क्रीनशॉट": "screenshot", "क्लिपबोर्ड": "clipboard",
    "फोटो": "photo", "पिक्चर": "picture", "कैलकुलेट": "calculate", "कैलकुलेटर": "calculator", "नोट": "note",
    "टास्क": "task", "टास्क्स": "tasks", "लिस्ट": "list", "मोड": "mode", "फॉर्मल": "formal", "विटी": "witty",
    "सेवेज": "savage", "डीप": "deep", "थिंक": "think", "स्टॉप": "stop", "कैंसिल": "cancel", "ओपन": "open",
    "क्लोज़": "close", "प्ले": "play", "पॉज़": "pause",
}
_LOAN_RE = re.compile("|".join(sorted(map(re.escape, _LOAN), key=len, reverse=True)))


def loanwords(text: str) -> str:
    """'सेल्स का फोरकास्ट दिखाओ' -> 'sales का forecast दिखाओ' (English technical words written in Devanagari)."""
    if not _DEV.search(text):
        return text
    out, last = [], 0
    for m in _LOAN_RE.finditer(text):
        a, b = m.start(), m.end()
        # whole words only: the neighbours must not be Devanagari letters or vowel signs
        if (a > 0 and _DEV.match(text[a - 1])) or (b < len(text) and _DEV.match(text[b])):
            continue
        out.append(text[last:a])
        out.append(_LOAN[m.group(0)])
        last = b
    out.append(text[last:])
    return "".join(out)


# ------------------------------------------------------------------ places

_CITIES = {
    "दिल्ली": "Delhi", "नई दिल्ली": "New Delhi", "मुंबई": "Mumbai", "बंबई": "Mumbai", "कोलकाता": "Kolkata",
    "चेन्नई": "Chennai", "बेंगलुरु": "Bengaluru", "बेंगलोर": "Bengaluru", "बैंगलोर": "Bengaluru", "बैंगलुरु": "Bengaluru",
    "हैदराबाद": "Hyderabad", "पुणे": "Pune", "अहमदाबाद": "Ahmedabad", "जयपुर": "Jaipur", "लखनऊ": "Lucknow",
    "कानपुर": "Kanpur", "नागपुर": "Nagpur", "इंदौर": "Indore", "भोपाल": "Bhopal", "पटना": "Patna", "सूरत": "Surat",
    "वडोदरा": "Vadodara", "चंडीगढ़": "Chandigarh", "चंडीगढ": "Chandigarh", "गुरुग्राम": "Gurgaon", "गुड़गांव": "Gurgaon",
    "नोएडा": "Noida", "गाज़ियाबाद": "Ghaziabad", "वाराणसी": "Varanasi", "बनारस": "Varanasi", "आगरा": "Agra",
    "अमृतसर": "Amritsar", "लुधियाना": "Ludhiana", "रांची": "Ranchi", "भुवनेश्वर": "Bhubaneswar", "गोवा": "Goa",
    "कोच्चि": "Kochi", "कोचीन": "Kochi", "तिरुवनंतपुरम": "Thiruvananthapuram", "मैसूर": "Mysore", "कोयंबटूर": "Coimbatore",
    "विशाखापत्तनम": "Visakhapatnam", "रायपुर": "Raipur", "देहरादून": "Dehradun", "शिमला": "Shimla", "श्रीनगर": "Srinagar",
    "जोधपुर": "Jodhpur", "उदयपुर": "Udaipur", "मेरठ": "Meerut", "प्रयागराज": "Prayagraj", "इलाहाबाद": "Prayagraj",
    "गुवाहाटी": "Guwahati", "लंदन": "London", "न्यूयॉर्क": "New York", "दुबई": "Dubai", "सिंगापुर": "Singapore",
    "पेरिस": "Paris", "टोक्यो": "Tokyo", "सिडनी": "Sydney", "टोरंटो": "Toronto", "बर्लिन": "Berlin",
    "सैन फ्रांसिस्को": "San Francisco", "वॉशिंगटन": "Washington", "बीजिंग": "Beijing", "काठमांडू": "Kathmandu",
}
_CITY_RE = re.compile("|".join(sorted(map(re.escape, _CITIES), key=len, reverse=True)))
_STOP_CITY = {"aaj", "abhi", "kal", "yahan", "yaha", "ghar", "bahar", "aaj ka", "mera", "mere", "aapka", "iska", "uska",
              "tumhara", "bahut", "kaisa", "kaisi"}


def _city_from(text: str):
    m = _CITY_RE.search(text)
    if m:
        return _CITIES[m.group(0)]
    # Roman Hinglish: "pune ka mausam", "mausam in pune", "delhi mein mausam"
    m = re.search(r"\b([a-z][a-z .'-]{1,24}?)\s+(?:ka|ki|ke|mein|me|main)\s+(?:mausam|weather|temperature|taapman)", text, re.I)
    if m and m.group(1).strip().lower() not in _STOP_CITY:
        return m.group(1).strip().title()
    m = re.search(r"\b(?:mausam|weather|temperature)\s+(?:in|at|for)\s+([a-z][a-z .'-]{1,24})", text, re.I)
    if m:
        return m.group(1).strip().title()
    return None


# ------------------------------------------------------------------ the commands

_DUR = re.compile(f"(\\d+(?:\\.\\d+)?)\\s*({_UNIT_ALT}){_NOT_AFTER}", re.I)
_TIMER_WORD = re.compile(f"{_NOT_BEFORE}(टाइमर|टाईमर|timer|अलार्म|alarm|taimar){_NOT_AFTER}", re.I)
_REMIND = re.compile(f"(याद\\s*दिला[{_W}]*|yaad\\s*dila[{_W}]*|reminder|रिमाइंडर|जगा[{_W}]*|jaga[{_W}]*|बता\\s*देना|bata\\s*dena|alert)", re.I)
_AFTER = re.compile(f"{_NOT_BEFORE}(के\\s*बाद|ke\\s*baad|बाद|baad|mein|me|में){_NOT_AFTER}", re.I)
_CANCEL = re.compile("(बंद|band|रद्द|radd|cancel|हटा|hata|delete|clear|ख़त्म|khatam|रोक)", re.I)
_FILLER_WORDS = re.compile(
    f"{_NOT_BEFORE}(मुझे|mujhe|मेरे|mere|मेरा|mera|को|ko|का|ka|की|ki|के|ke|एक|ek|कर|kar|करो|karo|दो|do|देना|dena|दीजिए|"
    f"dijiye|दे|de|प्लीज़|प्लीज|please|ज़रा|zara|जरा|jara|तुम|tum|आप|aap|जार्विस|jarvis|भाई|bhai|यार|yaar|है|hai|हो|ho|"
    f"सेट|set|लगाओ|lagao|लगा|laga|चालू|chalu|शुरू|shuru){_NOT_AFTER}", re.I)


def _duration_phrase(text: str):
    parts = [f"{m.group(1)} {_UNIT[m.group(2).lower()]}" for m in _DUR.finditer(text)]
    return " ".join(parts) if parts else None


def _clean_label(label: str) -> str:
    label = _FILLER_WORDS.sub(" ", label)
    label = re.sub(r"(के\s*लिए|ke\s*liye|liye|लिए|for)\s*$", " ", label.strip(), flags=re.I)
    return re.sub(r"\s+", " ", label).strip(" ,.?!।")


def _timer(t: str):
    dur = _duration_phrase(t)
    if not dur:
        if _TIMER_WORD.search(t) and _CANCEL.search(t):
            return "cancel all timers"
        return None
    is_remind = bool(_REMIND.search(t))
    if _TIMER_WORD.search(t) and not is_remind:
        if _CANCEL.search(t):
            return "cancel all timers"
        label = _clean_label(_TIMER_WORD.sub(" ", _DUR.sub(" ", t)))
        return f"set a {dur} timer" + (f" for {label}" if label else "")
    if is_remind and (_AFTER.search(t) or _TIMER_WORD.search(t)):
        label = _DUR.sub(" ", t)
        label = _REMIND.sub(" ", label)
        label = _AFTER.sub(" ", label)
        label = _TIMER_WORD.sub(" ", label)
        label = _clean_label(label)
        return f"remind me in {dur}" + (f" to {label}" if label else "")
    return None


def _weather(t: str, words: int):
    if words > 9 or not re.search(r"(मौसम|mausam|weather|temperature|तापमान|taapman|बारिश|baarish|barish)", t, re.I):
        return None
    if re.search(r"(मौसम|mausam|weather|तापमान|taapman|temperature)", t, re.I) is None:
        return "will it rain today"                      # "बारिश होगी क्या"
    city = _city_from(t)
    return f"what is the weather in {city}" if city else "what is the weather"


def _news(t: str, words: int):
    if words > 9 or not re.search(r"(खबर|ख़बर|खबरें|ख़बरें|khabar|khabrein|samachar|समाचार|headlines|हेडलाइन|\bnews\b)", t, re.I):
        return None
    if not re.search(r"(सुना|बता|दिखा|batao|bataiye|sunao|suna|dikhao|dikha|aaj|आज|ताज़ा|taaza|latest|नई|nayi|क्या|kya)", t, re.I):
        return None
    m = re.search(f"([{_W} .'-]{{2,40}}?)\\s+(?:की|ki|ke baare mein|के बारे में|पर|ke)\\s+(?:ताज़ा\\s+|taaza\\s+)?"
                  "(?:खबर|ख़बर|khabar|samachar|समाचार)", t, re.I)
    if m:
        topic = _FILLER_WORDS.sub(" ", m.group(1))
        topic = re.sub(r"(आज|aaj|latest|ताज़ा|taaza)", " ", topic)
        topic = re.sub(r"\s+", " ", topic).strip()
        if topic and len(topic) > 1:
            return f"news about {_CITIES.get(topic, topic)}"
    return "latest news"


def _simple(t: str, words: int):
    if words <= 7 and re.search(r"(बैटरी|battery)", t, re.I) and re.search(
            r"(कितनी|kitni|कितना|kitna|बची|bachi|चार्ज|charge|level|status|hai|है)", t, re.I):
        return "battery"
    if words <= 7 and re.search(r"(कितने\s*बजे|kitne\s*baje|टाइम\s*क्या|time\s*kya|समय\s*क्या|samay\s*kya|क्या\s*टाइम|kya\s*time)", t, re.I):
        return "what time is it"
    if words <= 8 and re.search(r"(आज\s*की\s*(?:तारीख|तारीख़|date)|aaj\s*(?:ki|kya)\s*(?:tarikh|date)|आज\s*कौन\s*सा\s*दिन|"
                                r"aaj\s*kaun\s*sa\s*din|date\s*kya|तारीख\s*क्या)", t, re.I):
        return "what's the date"
    return None


def _memory_tasks(t: str, words: int):
    m = re.match(r"^(?:जार्विस[, ]*)?(?:याद\s*रखो|याद\s*रखना|yaad\s*rakho|yaad\s*rakhna|yaad\s*rakh)\s*(?:कि|ki|ke|that)?\s*(.+)$", t, re.I)
    if m:
        return f"remember that {m.group(1).strip()}"
    if re.search(r"(मुझे\s*मेरे\s*बारे\s*में|mere\s*baare\s*mein|मेरे\s*बारे\s*में\s*क्या)", t, re.I) and re.search(
            r"(पता|याद|pata|yaad|jaante|जानते)", t, re.I):
        return "what do you remember about me"
    m = re.match(r"^(?:भूल\s*जाओ|भूल\s*जा|bhool\s*jao|bhul\s*jao|bhool\s*ja)\s*(.+)$", t, re.I)
    if m:
        return f"forget {m.group(1).strip()}"
    if words <= 8 and re.search(r"(टास्क|tasks?|to-?do|काम\s*की\s*लिस्ट|kaam\s*ki\s*list)", t, re.I) and re.search(
            r"(दिखा|बता|सुना|dikha|batao|sunao|show|list|कौन|kaun)", t, re.I):
        return "show tasks"
    m = re.match(r"^(?:मुझे|mujhe)\s+(.+?)\s+(?:याद\s*दिला\w*|yaad\s*dila\w*)$", t, re.I)
    if m and not _duration_phrase(t):
        return f"remind me to {m.group(1).strip()}"
    return None


_OPS = [(r"(?:प्लस|plus|जमा|jama|\+|add)", "+"), (r"(?:माइनस|minus|घटा|ghata|−)", "-"),
        (r"(?:गुणा|guna|into|times|multiply|\*|×)", "*"), (r"(?:भाग|bhag|divided by|÷)", "/")]


def _calc(t: str, words: int):
    if words > 9 or not re.search(r"(कितना\s*होता|kitna\s*hota|कितने\s*होते|kitne\s*hote|कितना\s*हुआ|kitna\s*hua|calculate|कैलकुलेट)", t, re.I):
        return None
    expr = re.sub(r"(कितना\s*होता\s*है|कितने\s*होते\s*हैं|कितना\s*हुआ|kitna\s*hota\s*hai|kitne\s*hote\s*hain|kitna\s*hua|"
                  r"calculate|कैलकुलेट|है|hai|\?)", " ", t, flags=re.I)
    for pattern, sym in _OPS:
        expr = re.sub(rf"\s{pattern}\s", f" {sym} ", f" {expr} ", flags=re.I)
    expr = re.sub(r"[^\d+\-*/().\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    if re.search(r"\d", expr) and re.search(r"[+\-*/]", expr):
        return f"calculate {expr}"
    return None


_HINGLISH_HINT = re.compile(
    r"\b(ka|ki|ke|ko|mein|baad|lagao|laga|karo|batao|bata|dikhao|dikha|sunao|suna|kitna|kitni|kitne|baje|kya|hai|yaad|"
    r"rakho|bhool|mausam|khabar|khabrein|ghante|ghanta|taaza)\b", re.I)


def normalize(text: str) -> str:
    """The English command behind a Hindi / Hinglish one, or the text itself when there is nothing to translate."""
    raw = (text or "").strip()
    if not raw:
        return raw
    t = re.sub(r"^\s*(?:hey\s+|ok\s+|okay\s+)?(?:jarvis|जार्विस|जारविस)\b[\s,.:!-]*", "", raw, flags=re.I).strip()
    t = t.rstrip("?.!।,").strip()
    has_dev = bool(_DEV.search(t))
    # only look at things that could be Hindi or Hinglish: plain English is never rewritten
    if not has_dev and not _HINGLISH_HINT.search(t):
        return raw

    num = numberize(t)
    words = len(num.split())
    loose = numberize(t, loose=True)
    for fn in (lambda: _timer(num), lambda: _weather(num, words), lambda: _news(num, words),
               lambda: _simple(num, words), lambda: _memory_tasks(num, words), lambda: _calc(loose, words)):
        out = fn()
        if out:
            return out
    return loanwords(raw) if has_dev else raw
