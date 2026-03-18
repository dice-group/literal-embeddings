import csv
import os


class KGEntTextDataset:
    """Dataset for entity-description triples stored in tab-separated text files."""

    def __init__(
        self,
        dataset_dir: str ,
        file_name,
        entity_to_idx,
        entity_names=None,
        relation_uri=None,
        allowed_entities=None,
        deduplicate=True,
        lowercase_text=False,
        strip_text=True,
    ):
        self.file_name = file_name
        self.file_path = os.path.join(dataset_dir, "literals", file_name)
        self.entity_to_idx = entity_to_idx
        self.entity_names = entity_names or {}
        self.relation_uri = relation_uri
        self.allowed_entities = set(allowed_entities) if allowed_entities is not None else None
        self.deduplicate = deduplicate
        self.lowercase_text = lowercase_text
        self.strip_text = strip_text
        self.samples = self._load_samples()


    def _normalize_text(self, text):
        if self.strip_text:
            text = text.strip()
        if self.lowercase_text:
            text = text.lower()
        return text

    def _load_samples(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Text literals file not found: {self.file_path}")

        samples = []
        seen = set()
        with open(self.file_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) != 3:
                    continue
                entity, relation, description = row
                entity_idx = self.entity_to_idx.get(entity)
                if entity_idx is None:
                    continue
                if self.relation_uri is not None and relation != self.relation_uri:
                    continue
                if self.allowed_entities is not None and entity not in self.allowed_entities:
                    continue

                description = self._normalize_text(description)
                if not description:
                    continue

                sample = {
                    "entity": entity,
                    "entity_name": self.entity_names.get(entity, entity),
                    "entity_idx": entity_idx,
                    "relation": relation,
                    "description": description,
                }
                sample_key = (entity, relation, description)
                if self.deduplicate and sample_key in seen:
                    continue
                seen.add(sample_key)
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
