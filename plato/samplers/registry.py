"""
The registry for samplers designed to partition the dataset across the clients.
Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from plato.samplers.mindspore import (
        iid as iid_mindspore,
        dirichlet as dirichlet_mindspore,
    )

    registered_samplers = OrderedDict([
        ('iid', iid_mindspore.Sampler),
        ('noniid', dirichlet_mindspore.Sampler),
    ])
elif hasattr(Config().trainer, 'use_tensorflow'):
    from plato.samplers.tensorflow import base
    registered_samplers = OrderedDict([
        ('iid', base.Sampler),
        ('noniid', base.Sampler),
        ('mixed', base.Sampler),
    ])
else:
    from plato.samplers import (iid, dirichlet, mixed, orthogonal,
                                all_inclusive)

    registered_samplers = OrderedDict([
        ('iid', iid.Sampler),
        ('noniid', dirichlet.Sampler),
        ('mixed', mixed.Sampler),
        ('orthogonal', orthogonal.Sampler),
        ('all_inclusive', all_inclusive.Sampler),
    ])


def get(datasource, client_id, testing=False):
    """Get an instance of the sampler."""
    if testing:
        if hasattr(Config().data, 'test_set_sampler'):
            sampler_type = Config().data.test_set_sampler
            logging.info("[Client #%d] Test set sampler: %s", client_id,
                         sampler_type)
    else:
        if hasattr(Config().data, 'sampler'):
            sampler_type = Config().data.sampler
        else:
            sampler_type = 'iid'

        logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](datasource,
                                                               client_id,
                                                               testing=testing)
    else:
        raise ValueError('No such sampler: {}'.format(sampler_type))

    return registered_sampler