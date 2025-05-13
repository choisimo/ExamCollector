import gettext
import os
from functools import lru_cache

class LocaleManager:
    def __init__(self, locale_dir: str, lang: str = "en_US"):
        self.locale_dir = locale_dir
        self.lang = lang
        self.trans = gettext.translation(
            "messages", localedir=locale_dir, languages=[lang], fallback=True
        )

    @lru_cache(maxsize=32)
    def translate(self, msgid: str) -> str:
        return self.trans.gettext(msgid)

    def switch(self, lang: str):
        self.lang = lang
        self.trans = gettext.translation(
            "messages", localedir=self.locale_dir, languages=[lang], fallback=True
        )
