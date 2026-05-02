import pandas as pd
import json
import re

class ProsesData:
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def _baca_file(self, file_name: str) -> str:
        full_path = f"{self.base_path}{file_name}"
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def _parsing_json(self, content: str) -> str:
        content_fixed = re.sub(r'\}\s*\{', '},{', content)
        return f"[{content_fixed}]"

    def muat_dataframe(self, file_name: str) -> pd.DataFrame:
        raw_content = self._baca_file(file_name)
        valid_json_string = self._parsing_json(raw_content)
        data = json.loads(valid_json_string)
        return pd.DataFrame(data)