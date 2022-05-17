from transformers import PfeifferConfig, HoulsbyConfig, ParallelConfig, CompacterConfig

ADAPTER_MAP = {
    'pfeiffer': PfeifferConfig,
    'houlsby': HoulsbyConfig,
    'parallel': ParallelConfig,
    'compacter': CompacterConfig,
}