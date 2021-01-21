import csv
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union, Tuple

import pandas as pd
import pyarrow as pa
from tqdm.auto import tqdm


class DataFrame(object):
    """A DataFrame based on pyarrow"""

    def __init__(self, arrow_table: pa.Table):
        self._data: pa.Table = arrow_table

    @classmethod
    def from_file(cls, filename: str) -> "DataFrame":
        mmap = pa.memory_map(filename)
        f = pa.ipc.open_stream(mmap)
        pa_table = f.read_all()
        return cls(arrow_table=pa_table)

    @classmethod
    def from_csv(
        cls,
        filename: str,
        fieldnames: Tuple[str],
        delimiter="\t",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
    ) -> "DataFrame":
        columnar_data = defaultdict(list)
        with open(filename, encoding="utf-8") as csv_file:
            for row in csv.DictReader(
                csv_file,
                quotechar=quotechar,
                delimiter=delimiter,
                quoting=quoting,
                skipinitialspace=skipinitialspace,
                fieldnames=fieldnames,
            ):
                for column, value in row.items():
                    columnar_data[column].append(value)
        return cls.from_dict(columnar_data)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrame":
        pa_table: pa.Table = pa.Table.from_pandas(df=df, schema=None)
        return cls(pa_table)

    @classmethod
    def from_dict(cls, mapping: dict) -> "DataFrame":
        pa_table: pa.Table = pa.Table.from_pydict(mapping=mapping)
        return cls(pa_table)

    def save_to_file(
        self, dataset_path: str, max_chunksize: Optional[int] = None
    ) -> None:
        stream = pa.OSFile(dataset_path, "wb")
        writer = pa.RecordBatchStreamWriter(stream, self._data.schema)
        batches: List[pa.RecordBatch] = self._data.to_batches(
            max_chunksize=max_chunksize
        )
        for batch in batches:
            writer.write_batch(batch)

    def map(
        self,
        function: Optional[Callable] = None,
        input_columns: Optional[Union[str, List[str]]] = None,
    ) -> "DataFrame":
        """
        apply function to each row in the dataframe

        # TODO:
        # 1. add options to support applying function to each batch
        # 2. use pa.Table.from_arrays() with chunked saving. Because from_dict() is slow and keep all data in memory.
        # 3. support multi-worker
        """
        assert len(self) > 0, "dataframe is empty!"
        processed_examples = []
        for i, example in enumerate(tqdm(self)):
            fn_args = (
                [example]
                if input_columns is None
                else [example[col] for col in input_columns]
            )
            processed_example = function(*fn_args)
            example.update(processed_example)
            processed_examples.append(example)
        assert len(processed_examples) > 0, "processed_examples is empty!"
        cols = sorted(processed_examples[0].keys())
        columnar = {}
        for col in cols:
            columnar[col] = [example[col] for example in processed_examples]
        return DataFrame.from_dict(columnar)

    @property
    def data(self) -> pa.Table:
        return self._data

    @property
    def num_rows(self) -> pa.Table:
        return self._data.num_rows

    def _getitem(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        if isinstance(key, int):
            if key < 0:
                key = self.num_rows + key
            if key >= self.num_rows or key < 0:
                raise IndexError(
                    f"Index ({key}) outside of table length ({self.num_rows})."
                )

            outputs = self._unnest(self._data.slice(key, 1).to_pydict())

        elif isinstance(key, slice):
            indices_array = None
            key_indices = key.indices(self.num_rows)

            # Check if the slice is a contiguous slice - else build an indices array
            if key_indices[2] != 1 or key_indices[1] < key_indices[0]:
                indices_array = pa.array(list(range(*key)), type=pa.uint64())

            # Get the subset of the table
            if indices_array is not None:
                data_subset = pa.concat_tables(
                    self._data.slice(indices_array[i].as_py(), 1)
                    for i in range(len(indices_array))
                )
            else:
                data_subset = self._data.slice(
                    key_indices[0], key_indices[1] - key_indices[0]
                )

            outputs = data_subset.to_pydict()

        elif isinstance(key, str):
            if key not in self._data.column_names:
                raise ValueError(
                    f"Column ({key}) not in table columns ({self._data.column_names})."
                )

            data_array = self._data.column(key)
            outputs = data_array.to_pylist()
        else:
            raise ValueError(
                "Can only get row(s) (int or slice or list[int]) or columns (string)."
            )
        return outputs

    @staticmethod
    def _unnest(py_dict):
        return dict((key, array[0]) for key, array in py_dict.items())

    @staticmethod
    def _nest(py_dict):
        return dict((key, [elem]) for key, elem in py_dict.items())

    def __del__(self):
        if hasattr(self, "_data"):
            del self._data

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        for index in range(self.num_rows):
            yield self._getitem(index)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        return self._getitem(key)
