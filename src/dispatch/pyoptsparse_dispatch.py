from .Dispatcher import Dispatcher
from .DispatchState import NumpyState
from utils import InputData
# from ..Components import Component
# from ..Cases import Case
from typing import List

import chickadee
import numpy as np
import sys
import os
import time as time_lib
import traceback

try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

def convert_dispatch(ch_dispatch: chickadee.DispatchState, resource_map: dict,
                      component_map: dict) -> NumpyState:
  '''Convert a Chickadee dispatch object to a NumpyState object
  @ In, ch_dispatch, chickadee Dispatch, The dispatch to convert
  @ In, resource_map, dict, HERON resource map
  @ In, component_map, dict, a map of Chickadee components to HERON ones
  @ Out, np_dispatch, NumpyState, The converted dispatch
  '''

  # This just needs to be a reliable unique identifier for each component.
  # In HERON it is a HERON Component. Here we just use the component names.
  np_dispatch = NumpyState()

  np_dispatch.initialize(component_map.values(), resource_map, ch_dispatch.time)

  start_i = 0
  end_i = len(ch_dispatch.time)

  # Copy over all the activities
  for c, data in ch_dispatch.state.items():
    for res, values in data.items():
      np_dispatch.set_activity_vector(component_map[c], res,
              start_i, end_i, values)

  return np_dispatch

def generate_transfer(comp, sources, dt):
  interaction = comp.get_interaction()
  if comp._stores:
    # For storage components there is no transfer, so the transfer method is attached to the 
    # rate node instead.
    transfer = interaction._rate._obj._module_methods[interaction._rate._sub_name]
    if transfer is None:
      raise Exception(f'A Storage component ({comp.name}) cannot be defined without a transfer function when using the Chickadee dispatcher.')
    return transfer
  else:
    thing = interaction._transfer
    if thing is None:
      # For components that don't actually transfer, HERON never loads the functions
      return lambda x: {}
  # We really need to dig for this one, but it lets us determine our own
  # function signatures for the transfer functions by bypassing the HERON interfaces
  transfer = interaction._transfer._obj._module_methods[thing._sub_name]
  return transfer

class PyOptSparse(Dispatcher):
  '''
  Dispatch using pyOptSparse optimization package through Chickadee
  '''

  @classmethod
  def get_input_specs(cls):
    specs = InputData.parameterInputFactory('pyoptsparse', ordered=False, baseNode=None)
    return specs

  def __init__(self):
    self.name = 'PyOptSparseDispatcher'

  # def dispatch(self, case: Case, components: List[Component], sources, meta):
  def dispatch(self, case, components, sources, meta):
    """
      Dispatch the system using IPOPT, pyOptSparse and Chickadee.
      @ In, case, Case,
      @ In, components, List[Component], the system components
      @ In, sources
      @ In, meta
      @ Out, opt_dispatch, NumpyState, the Optimal system dispatch
    """
    # Get the right time horizon
    time_horizon = np.linspace(*self.get_time_discr())
    dt = time_horizon[1] - time_horizon[0]

    resource_map = meta['HERON']['resource_indexer']

    # Convert the components to Chickadee components
    ch_comps = []
    comp_map = {}
    for c in components:
      tf = c.get_interaction()._transfer
      capacity_var = c.get_capacity_var()
      cap = c.get_capacity(meta)[0][capacity_var]
      capacity = np.ones(len(time_horizon)) * cap
      ch_comp = chickadee.PyOptSparseComponent(
        c.name,
        capacity,
        1e5*np.ones(len(time_horizon)),
        1e5*np.ones(len(time_horizon)),
        capacity_var,
        generate_transfer(c, sources, dt),
        None, # External cost function is used
        produces=list(c.get_outputs()),
        consumes=list(c.get_inputs()),
        # It turns out c._stores only holds a <HERON Storage> object,
        # so we get the resource from the inputs
        stores=list(c.get_inputs())[0] if c._stores else None,
        dispatch_type=c.is_dispatchable()
      )
      ch_comps.append(ch_comp)
      comp_map[ch_comp.name] = c

    # Make the objective function
    def objective(dispatchState: chickadee.DispatchState):
    #print(len(dispatchState.time), {key: { res: len(d) for res, d in dispatchState.state[key].items()}for key in dispatchState.state.keys()})
      np_dispatch = convert_dispatch(dispatchState, resource_map, comp_map)
      return self._compute_cashflows(components, np_dispatch,
                                      dispatchState.time, meta)

    # Dispatch using Chickadee
    dispatcher = chickadee.PyOptSparse()
    print('Chickadee components:', ch_comps)
    solution = dispatcher.dispatch(ch_comps, time_horizon, meta=meta,
                                        external_obj_func=objective)

    # Convert Chickadee dispatch back to HERON dispatch for return
    solution_dispatch = chickadee.DispatchState(ch_comps, time_horizon)
    solution_dispatch.state = solution.dispatch
    return convert_dispatch(solution_dispatch, resource_map, comp_map)
