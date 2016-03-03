from blocks.extensions import FinishAfter
from blocks.extensions.saveload import Checkpoint

class FinishIfNoImprovementAfterPlus(FinishAfter):
    """Stop after improvements have ceased for a dynamic period.

    Parameters
    ----------
    notification_name : str
        The name of the log record to look for which indicates a new
        best performer has been found.  Note that the value of this
        record is not inspected.
    patience_callback: callback
        Return the patience from the epoch.
    patience_log_record : str, optional
        The name under which to record the number of iterations we
        are currently willing to wait for a new best performer.
        Defaults to `notification_name + '_patience_epochs'` or
        `notification_name + '_patience_iterations'`, depending
        which measure is being used.

    Notes
    -----
    By default, runs after each epoch. This can be manipulated via
    keyword arguments (see :class:`blocks.extensions.SimpleExtension`).

    """
    def __init__(self, notification_name,
                 patience_log_record=None, **kwargs):
        self.notification_name = notification_name
        kwargs.setdefault('after_epoch', True)
        self.last_best_iter = self.last_best_epoch = None
        if patience_log_record is None:
            self.patience_log_record = notification_name + '_patience_epochs'
        else:
            self.patience_log_record = patience_log_record
        super(FinishIfNoImprovementAfterPlus, self).__init__(**kwargs)

    def update_best(self):
        # Here mainly so we can easily subclass different criteria.
        if self.notification_name in self.main_loop.log.current_row:
            self.last_best_iter = self.main_loop.log.status['iterations_done']
            self.last_best_epoch = self.main_loop.log.status['epochs_done']

    def do(self, which_callback, *args):
        self.update_best()
        # If we haven't encountered a best yet, then we should just bail.
        if self.last_best_iter is None:
            return

        since = (self.main_loop.log.status['epochs_done'] -
                 self.last_best_epoch)
        patience = self.patience_callback(self.main_loop.log.status['epochs_done']) - since

        self.main_loop.log.current_row[self.patience_log_record] = patience

        if patience <= 0:
            super(FinishIfNoImprovementAfterPlus, self).do(which_callback,
                                                       *args)

    def patience_callback(self, e):
        return max(int(e/2), 10)

class CheckpointBest(Checkpoint):

    def __init__(self, notification_name, path, **kwargs):
        self.notification_name = notification_name
        super(CheckpointBest, self).__init__(path, **kwargs)

    def do(self, which_callback, *args):
        if self.notification_name in self.main_loop.log.current_row:
            super(CheckpointBest, self).do(which_callback, *args)

