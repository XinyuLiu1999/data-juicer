from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import Filter

ray = LazyLoader("ray")
redis = LazyLoader("redis")

MERSENNE_PRIME = (1 << 61) - 1


class DedupSet:
    def __init__(self):
        self.hash_record = set()

    def is_unique(self, key):
        if key not in self.hash_record:
            self.hash_record.add(key)
            return True
        else:
            return False

    def are_unique(self, keys):
        """Batch check: returns list of bools in same order as keys."""
        results = []
        for key in keys:
            if key not in self.hash_record:
                self.hash_record.add(key)
                results.append(True)
            else:
                results.append(False)
        return results


def get_remote_dedup_set():
    """Get the remote version of DedupSet with Ray decorator applied at runtime.
    Uses num_cpus=0 so actors don't compete with map_batches tasks for CPU
    slots, which would cause deadlock when tasks call ray.get() on actors."""
    return ray.remote(num_cpus=0, scheduling_strategy="SPREAD")(DedupSet)


class Backend(ABC):
    """
    Backend for deduplicator.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_unique(self, md5_value: str):
        pass


class ActorBackend(Backend):
    """
    Ray actor backend for deduplicator.
    Uses lazy initialization to defer actor creation until first use,
    allowing the cluster to autoscale before actors consume resources.
    """

    def __init__(self, dedup_set_num: Union[int, str], RemoteDedupSet=None):
        # Store config but don't create actors yet
        # dedup_set_num can be int or "auto"
        self._dedup_set_num_config = dedup_set_num
        self._RemoteDedupSet = RemoteDedupSet
        self._dedup_sets = None  # Lazy - created on first use
        self._actual_dedup_set_num = None

    @property
    def dedup_set_num(self):
        """Get actual dedup_set_num, calculating from cluster resources if 'auto'."""
        if self._actual_dedup_set_num is None:
            if self._dedup_set_num_config == "auto":
                self._actual_dedup_set_num = max(1, int(ray.cluster_resources().get("CPU", 1) / 2))
            else:
                self._actual_dedup_set_num = int(self._dedup_set_num_config)
        return self._actual_dedup_set_num

    def _ensure_actors(self):
        """Create actors on first use when cluster has scaled."""
        if self._dedup_sets is None:
            RemoteDedupSet = self._RemoteDedupSet or get_remote_dedup_set()
            self._dedup_sets = [RemoteDedupSet.remote() for _ in range(self.dedup_set_num)]

    def is_unique(self, md5_value: str):
        self._ensure_actors()
        dedup_set_id = int.from_bytes(md5_value.encode(), byteorder="little") % MERSENNE_PRIME % self.dedup_set_num
        return ray.get(self._dedup_sets[dedup_set_id].is_unique.remote(md5_value))

    def are_unique_batched(self, hash_values):
        """Check uniqueness for a batch of hash values using batched actor calls.

        Groups hashes by their target actor, sends one batched remote call per
        actor, then reassembles results in the original order. This makes only
        O(num_actors) remote calls instead of O(num_hashes).
        """
        self._ensure_actors()
        num = self.dedup_set_num

        # Group (original_index, hash) by actor id
        actor_groups = defaultdict(list)
        for idx, hv in enumerate(hash_values):
            actor_id = (
                int.from_bytes(hv.encode(), byteorder="little")
                % MERSENNE_PRIME % num
            )
            actor_groups[actor_id].append((idx, hv))

        # One batched remote call per actor
        futures = []
        actor_ids = []
        for actor_id, items in actor_groups.items():
            keys = [hv for _, hv in items]
            futures.append(self._dedup_sets[actor_id].are_unique.remote(keys))
            actor_ids.append(actor_id)

        # Wait for all actors at once
        all_results = ray.get(futures)

        # Reassemble results in original order
        results = [None] * len(hash_values)
        for actor_id, batch_results in zip(actor_ids, all_results):
            items = actor_groups[actor_id]
            for (orig_idx, _), result in zip(items, batch_results):
                results[orig_idx] = result

        return results


class RedisBackend(Backend):
    """
    Redis backend for deduplicator.
    """

    def __init__(self, redis_address: str):
        self.redis_address = redis_address
        self.redis_client = redis.from_url(url=self.redis_address)
        self.redis_client.flushdb(0)

    def is_unique(self, md5_value: str):
        return self.redis_client.setnx(md5_value, 1)


class RayBasicDeduplicator(Filter):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    _batched_op = True

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"

    def __init__(
        self,
        backend: str = "ray_actor",
        redis_address: str = "redis://localhost:6379",
        dedup_set_num: Union[int, str] = "auto",
        *args,
        **kwargs,
    ):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param dedup_set_num: number of dedup set actors, or 'auto' to use CPU/2
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_address = redis_address
        self.backend = backend
        if backend == "ray_actor":
            # Pass dedup_set_num directly - ActorBackend handles "auto" lazily
            self.backend = ActorBackend(dedup_set_num)
        elif backend == "redis":
            # TODO: add a barrier to ensure that flushdb is performed before
            # the operator is called
            self.backend = RedisBackend(redis_address)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats_single(self, sample, context=False):
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_unique] = self.backend.is_unique(md5_value)
        return sample

    def compute_stats_batched(self, samples, context=False):
        keys = list(samples.keys())
        num_samples = len(samples[keys[0]])

        # Phase 1: calculate hashes for all samples in the batch
        hash_values = []
        for i in range(num_samples):
            this_sample = {key: samples[key][i] for key in keys}
            hash_values.append(self.calculate_hash(this_sample, context))

        # Phase 2: batch uniqueness checks
        if isinstance(self.backend, ActorBackend):
            results = self.backend.are_unique_batched(hash_values)
        else:
            results = [self.backend.is_unique(hv) for hv in hash_values]

        samples[HashKeys.is_unique] = results
        return samples

    def process_single(self, sample):
        return sample[HashKeys.is_unique]

    def process_batched(self, samples):
        return samples[HashKeys.is_unique]
