"""Compatibility object and static types for supported Lightning packages."""

from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import pytorch_lightning as _typed_lightning
    from pytorch_lightning.callbacks import Callback        as _TypedCallback
    from pytorch_lightning.callbacks import EarlyStopping   as _TypedEarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint as _TypedModelCheckpoint
    from pytorch_lightning.loggers   import CSVLogger       as _TypedCSVLogger
    from pytorch_lightning.loggers   import Logger          as _TypedLogger

    _lightning       = _typed_lightning
    _Callback        = _TypedCallback
    _CSVLogger       = _TypedCSVLogger
    _EarlyStopping   = _TypedEarlyStopping
    _Logger          = _TypedLogger
    _ModelCheckpoint = _TypedModelCheckpoint

    LightningModuleBase:     TypeAlias = _typed_lightning.LightningModule
    LightningDataModuleBase: TypeAlias = _typed_lightning.LightningDataModule
    TrainerType:             TypeAlias = _typed_lightning.Trainer
    CallbackBase:            TypeAlias = _TypedCallback
    LoggerType:              TypeAlias = _TypedLogger
else:
    try:
        import lightning.pytorch as _lightning
        from lightning.pytorch.callbacks import Callback        as _Callback
        from lightning.pytorch.callbacks import EarlyStopping   as _EarlyStopping
        from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint
        from lightning.pytorch.loggers   import CSVLogger       as _CSVLogger
        from lightning.pytorch.loggers   import Logger          as _Logger
    except ModuleNotFoundError:
        import pytorch_lightning as _lightning
        from pytorch_lightning.callbacks import Callback        as _Callback
        from pytorch_lightning.callbacks import EarlyStopping   as _EarlyStopping
        from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
        from pytorch_lightning.loggers   import CSVLogger       as _CSVLogger
        from pytorch_lightning.loggers   import Logger          as _Logger

    LightningModuleBase     = _lightning.LightningModule
    LightningDataModuleBase = _lightning.LightningDataModule
    TrainerType             = _lightning.Trainer
    CallbackBase            = _Callback
    LoggerType              = _Logger


class Lightning:
    """Expose one runtime adapter over ``lightning`` and ``pytorch_lightning``.

    LambdaForge prefers the modern ``lightning`` distribution but accepts an
    environment that provides the legacy package name. Framework modules use
    this object so the compatibility decision is made in exactly one place.
    """

    module          = _lightning
    Callback        = _Callback
    CSVLogger       = _CSVLogger
    EarlyStopping   = _EarlyStopping
    Logger          = _Logger
    ModelCheckpoint = _ModelCheckpoint
