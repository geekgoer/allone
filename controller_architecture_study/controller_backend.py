"""Controller configuration boundary shared by independent models and subnets."""
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class Configuration:
    key: int
    family: str
    name: str
    execution_id: int
    parameters: tuple = ()  # e.g. (('depth', 9), ('width', 512)) for a subnet


class ControllerBackend(Protocol):
    configurations: dict
    modes: dict

    def infer(self, key, raw237):
        """Return physical controls and measured timing for a configuration."""
        ...


class IP37Backend:
    """One whole non-preemptive request per independent deployed configuration."""
    def __init__(self, client, manifest, modes):
        self.client = client
        self.modes = {mode.numeric_id: mode for mode in modes}
        self.configurations = {
            item['numeric_id']: Configuration(item['numeric_id'], 'independent237', item['id'],
                                               item['numeric_id'])
            for item in manifest['models'] if item['numeric_id'] in self.modes}

    def infer(self, key, raw237):
        from .ip37_shadow import predict
        configuration = self.configurations[key]
        controls, timing = predict(self.client, configuration.execution_id, raw237)
        timing['configuration'] = configuration.key
        timing['controller_family'] = configuration.family
        return controls, timing
