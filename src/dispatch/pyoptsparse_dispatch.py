from .Dispatcher import Dispatcher
from .DispatchState import NumpyState
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

  def __gen_pool_cons(self, res):
    '''A closure for generating a pool constraint for a resource'''

    def pool_cons(dispatch_window: NumpyState):
      '''A resource pool constraint

      Ensures that the net amount of a resource being consumed, produced and
      stored is zero. Inteded to '''
      time = dispatch_window._times
      n = len(time)
      err = np.zeros(n)

      # FIXME: This is an inefficient way of doing this. Find a better way
      cs = [c for c in self.components if res in c.get_resources]
      for i, t in enumerate(time):
        for c in cs:
          err[i] += dispatch_window.get_activity(c, res, t)

      # FIXME: This simply returns the sum of the errors over time. There
      # are likely much better ways of handling this.
      return sum(err)

    return pool_cons

  def _build_pool_cons(self):
    '''Build the pool constraints
    Returns a list of `pool_cons` functions, one for each resource.'''

    cons = []
    for res in self.resources:
      # Generate the pool constraint here
      pool_cons = self.__gen_pool_cons(res)
      cons.append(pool_cons)
    return cons

  def _dispatch_pool(self, case, components, sources, meta):
    # Steps:
    #   1) Assemble all the vars into a vars dict
    #     A set of vars for each dispatchable component including storage elements
    #     include bound constraints
    #   2) Build the pool constraint functions
    #     Should have one constraint function for each pool
    #     Each constraint will be a function of all the vars
    #   3) Assemble the transfer function constraints
    #   4) Set up the objective function as the double integral of the incremental dispatch
    #   5) Assemble the parts for the optimizer function
    #     a) Declare the variables
    #     b) Declare the constraints
    #     c) Declare the objective function
    #     d) set the optimization configuration (IPOPT/SNOPT, CS/FD...)
    #   6) Run the optimization and handle failed/unfeasible runs
    #   7) Set the activities on each of the components and return the result

    print('\n\n\n\nPyOptDispatch:dispatch pool - components: ', components)
    print('case:', dir(case))
    print('sources:', sources)
    print('meta:', meta)

    self.components = components
    self.case = case
    self.resources = hutils.get_all_resources(components)

    # The time discretization for the dispatch is provided by the dispatch manager
    t_begin, t_end, t_num = self.get_time_discr()
    time = np.linspace(t_begin, t_end, t_num)
    print('time: ', time)
    resource_index_map = meta['HERON']['resource_indexer']

    #n = min(24, t_num) # FIXME: should find a way of getting this from the user
    # FIXME: Will need to implement rolling window dispatch soon

    # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
    vs = {} # Min/Max tuples of the various input
    for c in components:
      if c.is_dispatchable() == 'fixed':
        # Fixed dispatch components do not contribute to the variables
        continue
      else: # Independent and dependent dispatch
        vs[c.name] = {}
        cap_res = c.get_capacity_var()
        capacity = c.get_capacity(None, None, None, None)
        # print('capacity', c.name, cap_res, capacity)
        vs[c.name] = [0, capacity[0][cap_res]] # FIXME: get real min capacity
        vs[c.name].sort() # The max capacities are often negative
        # print('capacity bounds are:', vs[c.name])

    # Step 2) Build the resource pool constraint functions
    print('Step 2) Build the resource pool constraints')
    pool_cons = self._build_pool_cons()

    # Step 3) Assemble the transfer function constraints
    trans_cons = [] # FIXME: actually do it for real...

    # Step 4) Set up the objective function as the doible integral of the incremental dispatch
    print('Step 4) Assembling the big function')
    def obj(stuff):
      inner_meta = meta
      # Initialize the dispatch
      dispatch = NumpyState()
      dispatch.initialize(components, resource_index_map, time)

      # Dispatch the fixed components
      fixed_comps = [c for c in components if c.is_dispatchable() == 'fixed']
      for f in fixed_comps:
        resource = f.get_capacity_var()
        capacity = f.get_capacity(None, None, None, None)[0][resource]
        vals = np.ones(len(time)) * capacity
        # FIXME: Update the time indexes for rolling window dispatch
        dispatch.set_activity_vector(f, f.get_capacity_var(), 0, len(time), vals)

      # Dispatch the independent and dependent components using the vars
      disp_comps = [c for c in components if c.is_dispatchable() != 'fixed']
      for d in disp_comps:
        inter = d.get_interaction()
        # FIXME: Update the time indexes for rolling window dispatch
        for i in range(len(time)):
          request = {d.get_capacity_var(): stuff[d.name][i]}
          bal, inner_meta = inter.produce(request, inner_meta, sources, dispatch, i)
          for res, value in bal.items():
            dispatch.set_activity_indexed(d, resource_index_map[d][res], i, value)

      # At this point the dispatch should be fully determined, so assemble the return object
      things = {}
      # Dispatch the components to generate the obj val
      # FIXME: should probably use inner_meta here?
      things['objective'] = self._compute_cashflows(self.components, dispatch, time, meta)
      # Run the transfer function constraints
      things['transfer_functions'] = 0 # FIXME
      # Run the resource pool constraints
      things['resource_balance'] = [cons(dispatch) for cons in pool_cons]
      return things

    # Step 5) Assemble the parts for the optimizer function
    print('Step 5) Setting up pyOptSparse')
    optProb = pyoptsparse.Optimization(case.name, obj)
    for comp in vs.keys():
      for comp, bounds in vs.items():
        # FIXME: will need to find a way of generating the guess values
        optProb.addVarGroup(comp, t_num, 'c', lower=bounds[0], upper=bounds[1])
    optProb.addConGroup('resource_balance', len(pool_cons), lower=0, upper=0)
    optProb.addConGroup('tansfer_functions', len(trans_cons), lower=0, upper=0)
    optProb.addObj('objective')

    # Step 6) Run the optimization
    opt = pyoptsparse.OPT('IPOPT')
    sol = opt(optProb, sens='CD')
    print(sol)
    # FIXME: Raise error if failed to converge

    # Step 7) Set the activities on each component
    optDispatch = NumpyState()
    optDispatch.initialize(components, resource_index_map, time)
    for c in components:
      # optDispatch.set_activity_vector(c, res, start_time, end_time, vals)
      pass


    print('\nSuccessfully completed dispatch\n\n\n')
    return optDispatch


  def dispatch(self, case, components, sources, meta):
    return self._dispatch_pool(case, components, sources, meta)
    # return self._dispatch_tree(case, components, sources, meta)

# Questions:
# 3. How should I raise exceptions? What is the raven way?

