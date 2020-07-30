from .Dispatcher import Dispatcher
from pprint import pprint
from typing import List
import pyoptsparse
import numpy as np
import sys
import os

try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

class PyOptSparse(Dispatcher):
  '''
  Dispatch using pyOptSparse optimization package
  '''

  @classmethod
  def get_input_specs(cls):
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    return specs

  def __init__(self):
    print('PyOptDispatch:__init__')
    self.name = 'PyOptSparseDispatcher'
    self._window_len = 24

    # Defined on call to self.dispatch
    self.components = None
    self.case = None

  def read_input(self, specs):
    print('\n\n\n\nPyOptDispatch:read_input - specs: ', specs)

  def _build_pools(self):
    '''Assemble the resource pools including producing, consuming
    and storing components. Only used for pool method.
    Cannot be called before self.dispatch'''
    pool_dict = {
      'producers': [],
      'consumers': [],
      'storers': []
    }
    pools = {r: pool_dict for r in self.resources}

    for res, pool in pools.items():

      for c in self.components:
        if res in c.get_inputs():
          pool['consumers'].append(c)
        if res in c.get_outputs():
          pool['producers'].append(c)
        # FIXME: Add tracking for storers
    return pools

  def _build_graph(self):
    '''
    Build a graph of the components and the connections between them
    based on the resources consumed, stored or produced.
    '''
    graph = {c.name: [] for c in self.components}

    # Iterate over the resources
    for r in self.resources:
      producers = []
      consumers = []
      storers = []
      # make a list of the producers and consumers for the resource
      for c in self.components:
        if r in c.get_inputs():
          consumers.append(c)
        if r in c.get_outputs():
          producers.append(c)
        # FIXME: Add a method for handling storage

      # Make connections in the graph
      for s in storers:
        graph[s.name].append(s.name)
      for p in producers:
        for c in consumers:
          graph[p.name].append(c.name)
    return graph

  def _get_component(self, name: str):
    for c in self.components:
      if c.name == name:
        return c
    return

  def _find_var_groups(self):
    '''Find variable groups for the optimizer.
    These variables, together with any fixed dispatch should fully-
    specify the state of the system.
    '''
    vs = {}

    for c in self.components:
      if c.is_dispatchable() == 'independent':
        # Independently dispatched components are automatically vars
        # FIXME: Set better upper/lower bounds on the var group
        # Note that they could be given by a ValuedParam
        vs[c.name] = [0, 1]

      consumers = [self._get_component(comp) for comp in self.graph[c.name]]
      for p in c.get_outputs():
        iNum = 0
        res_cons = list(filter(lambda x: p in x.get_inputs(), consumers))
        for _ in range(len(res_cons) - 1):
          vs[f'{c.name}_{p}_{iNum}'] = [0, 1]
          # FIXME: Add constraints to ensure that the resource is conserved
          # accross the routing point
          iNum += 1
    return vs

  def _find_dispatchable(self, activities, to_dispatch):
    '''Find a component that is ready to be dispatched.
    This means that all the necessary inputs have been determined.'''

    for c in to_dispatch:
      # First dispatch those that have no inputs
      if len(c.get_inputs()) > 1:
        return c

      # Next check to see if all the input activities are supplied
      c_act = activities[c.name]
      supplied = True
      for r in c.get_inputs():
        if not c_act[r]:
          supplied = False
      if supplied:
        return c

    # FIXME: Should raise an error if only non-dispatchable components remain.

  def build_simulation(self):
    '''Build the simulation function. It determines all the activities of all
    components based on the variable groups.'''
    def simulation(x, meta, time):
      # Get the list of components to dispatch
      to_dispatch = self.components.copy()

      # Initialize the activity arrays
      activities = {}
      for c in self.components:
        activities[c.name] = {r: [] for r in c.get_resources()}

      # Dispatch the independent components
      # FIXME: Figure this part out or handle it below.
      # for c in to_dispatch:
      #   if c.is_dispatchable() == 'independent':
      #     activities[c.name] =

      while to_dispatch:
        # Get a component to dispatch
        c = self._find_dispatchable(activities, to_dispatch)


        # FIXME: This should never happen once the full dispatch is in place!
        if not c:
          break

        print(f'Dispatching {c.name}...')

        if c.is_dispatchable() == 'independent':
          for key in x[c.name]:
            activities[c.name][key] = x[c.name][key]
        elif c.is_dispatchable() == 'fixed':
          for key in meta[c.name]:
            activities[c.name][key] = meta[c.name][key]

        if c.get_outputs():
          # FIXME: run transfer function
          if c.get_inputs():
            inputs = {key: val for (key, val) in activities[c.name].items() if key in c.get_inputs()}

          # FIXME: assign activities to downstream components

        to_dispatch.remove(c)

      return activities
    return simulation

  def _obj_pool(self, x, components, meta):
    activities = {c.name: {} for c in components}

    for c in components:
      if c.is_dispatchable == 'fixed':
        # FIXME: Pull the dispatch from the component
        activities[c.name][c.cap_res] = 100

    # FIXME: Will likely need to turn this into a closure anyway to encapsulate the scope

  def _dispatch_pool(self, case, components, sources, meta):
    print('\n\n\n\nPyOptDispatch:dispatch pool - components: ', components)
    print('case:', case)
    print('sources:', sources)
    print('meta:', meta)

    self.components = components
    self.case = case
    self.resources = hutils.get_all_resources(components)
    n = 25  # number of time steps
    # This will be replaced with info from Heron once it is implemented
    t = np.linspace(0, 24, n)

    # Find the vars - 1 for each component input where dispatch is not fixed
    vs = {} # Min/Max tuples of the various input
    for c in components:
      if c.is_dispatchable() == 'fixed':
        # Fixed dispatch components do not contribute to the variables
        continue
      else: # Independent and dependent dispatch
        vs[c.name] = {}
        for r in c.get_inputs():
          vs[c.name][r] = (c.produce_min, c.produce_max)

    # Dispatch all the components - assume inputs are supplied, calculate outputs


    # Add constraints for each resources - net must be zero
    # Optimize subject to constraints
    # Return the optimal dispatch

    c = components[1]
    print(dir(c))
    print(f'Dispatching {c.name}...')
    # transfer = c.get_interaction().get_transfer()
    # print(transfer, transfer.type, dir(transfer))
    print(c.produce([50], meta, sources, [50], [1]))



  def _dispatch_tree(self, case, components, sources, meta):
    self.components = components
    self.case = case
    self.resources = hutils.get_all_resources(components)

    n  = 25 # number of time steps
    # This will be replaced with info from Heron once it is implemented
    time = np.linspace(0, 24, n)

    print('\n\n\n\nPyOptDispatch:dispatch tree - components: ', components)

    # The Pool-based method
    # # Build the pools
    # self.pools = self._build_pools()
    # print('Pools:', self.pools)

    # Graph-based method
    self.graph  = self._build_graph()
    print('graph: ', self.graph)
    print('var_groups:', self._find_var_groups())

    print('\nsources', type(sources), sources)
    print('\nvariables', type(meta), meta)

    sim = self.build_simulation()

    meta = {
      'steamer': {
        'steam': np.ones(n) * 100
      }
    }

    meta['generator'] = {
      'steam': np.ones(n)*100
    }

    c = components[0]


    # activities = sim(variables, meta, time)
    # print(activities)

  def dispatch(self, case, components, sources, meta):
    return self._dispatch_pool(case, components, sources, meta)
    # return self._dispatch_tree(case, components, sources, meta)






    print('\nSuccessfully completed dispatch\n\n\n')
    return

# Questions:
# 1. dispatch sets members on self that are later used by _get_resources and _build_graph
#   Is that ok or should I pass the relevant vars around instead?
# 3. How should I raise exceptions? What is the raven way?
# 4. Why do non-independent dispatched components often have a

# Notes:
# For fixed-dispatch are
