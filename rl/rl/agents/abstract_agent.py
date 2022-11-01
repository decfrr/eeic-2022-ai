from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from six import with_metaclass
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import os


class Agent(with_metaclass(ABCMeta, object)):
    """Abstract agent class."""

    @abstractmethod
    def act_and_train(self, obs, reward, done):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode_and_train(self, state, reward, done=False):
        """Observe conseqences and prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname):
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname):
        """Load internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_statistics(self):
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass
