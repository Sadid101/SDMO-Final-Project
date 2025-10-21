import csv
import json
import os.path
import pathlib

import numpy as np

from typing import Any
from copy import deepcopy

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self, tensor=None, /):

        self._tensor: np.ndarray = None
        self._vertices: list = []
        self._timestamps: list = []
        self._vertex_index_mapping: dict[Any, int] = {}
        self._timestamp_index_mapping: dict[Any, int] = {}

        if tensor is not None:
            match tensor:
                case np.ndarray():
                    if len(tensor.shape) != 3:
                        raise ValueError(f"SnapshotNetwork can be initialized from a 3D array, received {len(tensor.shape)}D")
                    if tensor.shape[0] != tensor.shape[1]:
                        raise ValueError(f"SnapshotNetwork can be initialized from an (N,N,T) 3D array, received array with shape {tensor.shape}")
                    N = tensor.shape[0]
                    T = tensor.shape[2]
                    self._tensor = tensor.copy()
                    self._vertices = list(range(N))
                    self._timestamps = list(range(T))
                    self._vertex_index_mapping = {i: i for i in range(N)}
                    self._timestamp_index_mapping = {i: i for i in range(N)}
                case SnapshotGraph():
                    self._tensor = tensor._tensor.copy()
                    self._vertices = deepcopy(tensor._vertices)
                    self._timestamps = deepcopy(tensor._timestamps)
                    self._vertex_index_mapping = {value: index for index, value in enumerate(self._vertices)}
                    self._timestamp_index_mapping = {value: index for index, value in enumerate(self._timestamps)}
                case _:
                    raise TypeError(f"Snapshot graph cannot be constructed from an object of type {type(tensor)}")


    @property
    def tensor(self):
        return self._tensor

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_value):
        assert type(new_value) is list
        assert len(self._vertices) == len(new_value)
        self._vertices = new_value
        self._vertex_index_mapping = {value: index for index, value in enumerate(new_value)}


    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, new_value):
        assert type(new_value) is list
        assert len(self._timestamps) == len(new_value)
        self._timestamps = new_value
        self._timestamp_index_mapping = {value: index for index, value in enumerate(new_value)}

    @property
    def N(self):
        return len(self._vertices)

    @property
    def T(self):
        return len(self._timestamps)

    @property
    def E(self):
        return np.count_nonzero(self._tensor)

    def permute_timestamps(self, new_order):
        assert type(new_order) is list
        assert set(new_order) == set(self._timestamps)
        permutation = [self._timestamp_index_mapping[timestamp] for timestamp in new_order]
        self._tensor = self._tensor[:, :, permutation]
        self._timestamps = new_order
        self._timestamp_index_mapping = {value: index for index, value in enumerate(new_order)}

    def permute_vertices(self, new_order):
        assert type(new_order) is list
        assert set(new_order) == set(self._vertices)
        permutation = [self._vertex_index_mapping[vertex] for vertex in new_order]
        self._tensor = self._tensor[permutation, :, :][:, permutation, :]
        self._vertices = new_order
        self._vertex_index_mapping = {value: index for index, value in enumerate(new_order)}

    def rename_vertex(self, old_vertex, new_vertex):
        i = self._vertex_index_mapping[old_vertex]
        self._vertices[i] = new_vertex
        self._vertex_index_mapping[new_vertex] = i
        del self._vertex_index_mapping[old_vertex]

    def rename_timestamp(self, old_timestamp, new_timestamp):
        i = self._timestamp_index_mapping[old_timestamp]
        self._timestamps[i] = new_timestamp
        self._timestamp_index_mapping[new_timestamp] = i
        del self._timestamp_index_mapping[old_timestamp]

    def vertex_lifespan(self, *vertex):
        lifespans = []
        for v in vertex:
            i = self._vertex_index_mapping[v]
            # Check for nonzero values along axis 1 or axis 2
            nonzero_mask = np.any(self._tensor[i, :, :] != 0, axis=0) | np.any(self._tensor[:, i, :] != 0, axis=0)  # Shape (T,)

            # Find the first and last nonzero indices
            first_t = np.argmax(nonzero_mask)  # First True index
            last_t = len(nonzero_mask) - 1 - np.argmax(nonzero_mask[::-1])  # Last True index
            lifespans.append((first_t, last_t))

        return lifespans

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 3:
            i, j, t = index
            if i not in self._vertex_index_mapping:
                raise IndexError(f"Source node {i} not found in graph")
            if j not in self._vertex_index_mapping:
                raise IndexError(f"Target node {j} not found in graph")
            if t not in self._timestamp_index_mapping:
                raise IndexError(f"Timestamp {t} not found in graph")
            i = self._vertex_index_mapping[i]
            j = self._vertex_index_mapping[j]
            t = self._timestamp_index_mapping[t]

            return self._tensor[i,j,t]
        else:
            raise TypeError("Index must consist of (source, target, timestamp)")

    def __setitem__(self, index, value):
        if not isinstance(value, float) or value < 0.0:
            raise ValueError(f"Weight must be a non-negative float, got {value}")

        if isinstance(index, tuple) and len(index) == 3:
            i, j, t = index
            if i not in self._vertex_index_mapping:
                raise IndexError(f"Source node {i} not found in graph")
            if j not in self._vertex_index_mapping:
                raise IndexError(f"Target node {j} not found in graph")
            if t not in self._timestamp_index_mapping:
                raise IndexError(f"Timestamp {t} not found in graph")
            i = self._vertex_index_mapping[i]
            j = self._vertex_index_mapping[j]
            t = self._timestamp_index_mapping[t]

            self._tensor[i,j,t] = value
        else:
            raise TypeError("Index must consist of (source, target, timestamp)")


    def load_edge_list(self, edge_list, vertex_list, timestamp_list,
                       directed=True, dtype=np.float32):
        vertex_index_mapping = {value: index for index, value in enumerate(vertex_list)}
        timestamp_index_mapping = {value: index for index, value in enumerate(timestamp_list)}
        max_vertex = len(vertex_list)
        max_time = len(timestamp_list)
        tensor = np.full((max_vertex, max_vertex, max_time), 0.0, dtype=dtype)
        for i, j, t, w in edge_list:
            i = vertex_index_mapping[i]
            j = vertex_index_mapping[j]
            t = timestamp_index_mapping[t]
            w = float(w)
            tensor[i, j, t] = w
            if directed is False:
                tensor[j, i, t] = w
        self._tensor = tensor
        self._vertices = vertex_list
        self._timestamps = timestamp_list
        self._vertex_index_mapping = vertex_index_mapping
        self._timestamp_index_mapping = timestamp_index_mapping

    def load_csv(self, csv_file, /, *, source='source', target='target', timestamp='timestamp', weight='weight',
                       directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):
        self.load_csv_list([csv_file], source=source, target=target, timestamp=timestamp, weight=weight,
                           directed=directed, dtype=dtype, sort_timestamps=sort_timestamps, sort_vertices=sort_vertices)

    def load_csv_directory(self, csv_dir, /, *, source='source', target='target', weight='weight',
                           directed=True, dtype=np.float32, sort_vertices=False):
        csv_files = [file for file in sorted(os.listdir(csv_dir))]
        self.load_csv_list(csv_files, source=source, target=target, weight=weight, directed=directed,
                           dtype=dtype, sort_vertices=sort_vertices, timestamp=None, sort_timestamps=False)

    def load_csv_list(self, csv_list, /, *, source='source', target='target',
                      timestamp='timestamp', weight='weight',
                      directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):

        rows = []
        vertex_set = set()
        timestamp_set = set()
        vertex_list = []
        timestamp_list = []
        for input_file in csv_list:
            with open(input_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_ = row[source]
                    target_ = row[target]
                    timestamp_ = str(pathlib.Path(input_file).with_suffix('')
                                     ) if timestamp is None else row[timestamp]
                    if source_ not in vertex_set:
                        vertex_set.add(source_)
                        vertex_list.append(source_)
                    if target_ not in vertex_set:
                        vertex_set.add(target_)
                        vertex_list.append(target_)
                    if timestamp_ not in timestamp_set:
                        timestamp_set.add(timestamp_)
                        timestamp_list.append(timestamp_)
                    rows.append([source_, target_, timestamp_, row.get(weight, 1.0)])

        if sort_vertices: vertex_list.sort()
        if sort_timestamps: timestamp_list.sort()
        self.load_edge_list(rows, vertex_list, timestamp_list, directed, dtype)

    def write_csv(self, path, /, *, source='source', target='target',
                  timestamp='timestamp', weight='weight'):

        csv_header = [source, target, timestamp, weight]
        csv_rows = []

        for source_vertex, i in self._vertex_index_mapping.items():
            for target_vertex, j in self._vertex_index_mapping.items():
                for timestamp_, t in self._timestamp_index_mapping.items():
                    if (w := self._tensor[i, j, t]) != 0.0:
                        csv_rows.append([source_vertex, target_vertex, timestamp_, w])

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
            writer.writerows(csv_rows)

    def load_json(self, json_file, /, *, source='source', target='target', timestamp='timestamp',
                    weight='weight', nodes='nodes', edges='edges',
                 directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):
        self.load_json_list([json_file], source=source, target=target, timestamp=timestamp, weight=weight,
                            nodes=nodes, edges=edges,
                            directed=directed, dtype=dtype, sort_timestamps=sort_timestamps, sort_vertices=sort_vertices)

    def load_json_directory(self, json_dir, /, *, source='source', target='target',
                        weight='weight', nodes='nodes', edges='edges',
                        directed=True, dtype=np.float32,
                        sort_vertices=False):

        json_files = [file for file in sorted(os.listdir(json_dir))]
        self.load_json_list(json_files, source=source, target=target, weight=weight, directed=directed,
                            nodes=nodes, edges=edges, timestamp=None,
                            dtype=dtype, sort_vertices=sort_vertices, sort_timestamps=False)

    def load_json_list(self, json_list, /, *, source='source', target='target',
                       timestamp='timestamp', weight='weight', nodes='nodes', edges='edges',
                       directed=True, dtype=np.float32,
                       sort_vertices=False, sort_timestamps=False):

        declared_vertex_set = set()
        seen_vertex_set = set()
        timestamp_set = set()
        vertex_list = []
        timestamp_list = []
        rows = []
        for file in json_list:
            with open(file, 'r') as f:
                data = json.load(f)
            declared_vertex_set.update(data[nodes])
            for edge in data[edges]:
                source_ = edge[source]
                target_ = edge[target]
                weight = edge.get(weight, 1.0)
                timestamp_ = str(pathlib.Path(file).with_suffix('')
                                 ) if timestamp is None else timestamp
                rows.append([source_, target_, timestamp_, weight])
                if source_ not in seen_vertex_set:
                    seen_vertex_set.add(source_)
                    vertex_list.append(source_)
                if target_ not in seen_vertex_set:
                    seen_vertex_set.add(target_)
                    vertex_list.append(target_)
                if timestamp_ not in timestamp_set:
                    timestamp_set.add(timestamp_)
                    timestamp_list.append(timestamp_)

        not_declared = seen_vertex_set - declared_vertex_set
        if not_declared:
            print(f'Following vertices were not declared in "{nodes}"', not_declared)

        not_connected = declared_vertex_set - seen_vertex_set
        if not_connected:
            print(f'Following vertices are not seen in "{edges}", and are ignored:', not_declared)


        if sort_vertices: vertex_list.sort()
        if sort_timestamps: timestamp_list.sort()
        self.load_edge_list(rows, vertex_list, timestamp_list, directed, dtype)

    def write_json(self, path, /, *, source='source', target='target', timestamp='timestamp',
                  weight='weight', nodes='nodes', edges='edges'):

        edge_data = []

        for source_vertex, i in self._vertex_index_mapping.items():
            for target_vertex, j in self._vertex_index_mapping.items():
                for timestamp_, t in self._timestamp_index_mapping.items():
                    if (w := self._tensor[i, j, t]) != 0.0:
                        edge_data.append({
                            source: source_vertex,
                            target: target_vertex,
                            weight: w,
                            timestamp: timestamp_
                        })
        json_data = {nodes: self._vertices, edges: edge_data}

        with open(path, 'w') as f:
            json.dump(json_data, f, indent=4)
