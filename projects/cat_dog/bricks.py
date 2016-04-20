from blocks.extensions import FinishAfter
#from blocks.extensions.saveload import Checkpoint

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

import os.path
import logging

from six.moves import cPickle

from blocks.extensions import SimpleExtension
from blocks.serialization import (
    secure_dump, DEFAULT_PROTOCOL)

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"

# Copy of Checkpoints with get save added
class Save(SimpleExtension):
    """Saves a pickled version of the main loop to the disk.

    The pickled main loop can be later reloaded and training can be
    resumed.

    Makes a `SAVED_TO` record in the log with the serialization destination
    in the case of success and ``None`` in the case of failure. The
    value of the record is a tuple of paths to which saving was done
    (there can be more than one if the user added a condition
    with an argument, see :meth:`do` docs).

    Parameters
    ----------
    path : str
        The destination path for pickling.
    save_separately : list of str, optional
        The list of the main loop's attributes to be pickled separately
        to their own files. The paths will be formed by adding
        the attribute name preceded by an underscore before the
        `path` extension. The whole main loop will still be pickled
        as usual.
    use_cpickle : bool
        See documentation of :func:`~blocks.serialization.dump`.

    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.


    """
    def __init__(self, path, save_separately=None, use_cpickle=False,
                 **kwargs):
        kwargs.setdefault("after_training", True)
        super(Save, self).__init__(**kwargs)
        if not save_separately:
            save_separately = []
        self.path = path
        self.save_separately = save_separately
        self.use_cpickle = use_cpickle

    def save_separately_filenames(self, path):
        """Compute paths for separately saved attributes.

        Parameters
        ----------
        path : str
            Path to which the main checkpoint file is being saved.

        Returns
        -------
        paths : dict
            A dictionary mapping attribute names to derived paths
            based on the `path` passed in as an argument.

        """
        root, ext = os.path.splitext(path)
        return {attribute: root + "_" + attribute + ext
                for attribute in self.save_separately}

    def get_save(self):
        return self.main_loop

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            secure_dump(self.get_save(), path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump,
                            protocol=DEFAULT_PROTOCOL)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
class SaveBest(Save):

    def __init__(self, notification_name, path, **kwargs):
        self.notification_name = notification_name
        kwargs.setdefault("after_epoch",True)
        super(SaveBest, self).__init__(path, **kwargs)

    def get_save(self):
        return self.main_loop.algorithm._cost_computation_graph

    def do(self, which_callback, *args):
        if self.notification_name in self.main_loop.log.current_row:
            super(SaveBest, self).do(which_callback, *args)

import time
class FinishAfterTime(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, seconds, **kwargs):
        kwargs.setdefault("after_epoch",True)
        self.seconds = seconds
        self.start = time.time()
        super(FinishAfterTime, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        if self.start + self.seconds < time.time():
            self.main_loop.log.current_row['training_finish_requested'] = True
