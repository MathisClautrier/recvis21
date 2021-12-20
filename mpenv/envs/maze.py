import os
import numpy as np
from gym import spaces
import hppfcl
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

from mpenv.core.mesh import Mesh
from mpenv.envs.base import Base
from mpenv.envs.maze_generator import Maze
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver
from mpenv.observers.maze import MazeObserver


class MazeGoal(Base):
    def __init__(self, grid_size):
        super().__init__(robot_name="sphere")

        self.thickness = 0.02
        self.grid_size = grid_size
        self.robot_name = "sphere"
        self.freeflyer_bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.robot_props = ROBOTS_PROPS["sphere2d"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.fig, self.ax, self.pos = None, None, None
        self.simple_like = False

    def _reset(self, idx_env=None, start=None, goal=None):
        model_wrapper = self.model_wrapper
        self.robot = self.add_robot("sphere2d", self.freeflyer_bounds)
        self.geoms, self.idx_env = self.get_obstacles_geoms(idx_env)
        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        model_wrapper.create_data()

        valid_sample = False
        while not valid_sample:
            self.state = self.random_configuration()
            if not self.simple_like:
                self.goal_state = self.random_configuration()
            else:
                cell = self.pick_goal_cell()
                ix,iy = (cell//self.grid_size +np.random.random())/self.grid_size,\
                (cell%self.grid_size +np.random.random())/self.grid_size
                q = np.zeros(7)
                q[0],q[1],q[-1] = ix,iy,1
                self.set_goal_state(q)
            valid_sample = self.validate_sample(self.state, self.goal_state)
        if start is not None:
            self.set_state(start)
        if goal is not None:
            self.set_goal_state(goal)

        if self.fig:
            plt.close()
        self.fig, self.ax, self.pos = None, None, None

        return self.observation()

    def validate_sample(self, state, goal_state):
        "Filter start and goal with straight path solution"
        straight_path = self.model_wrapper.arange(
            state, goal_state, self.delta_collision_check
        )
        _, collide = self.stopping_configuration(straight_path)
        return collide.any()
    
    def turn_simple_like(self):
        self.simple_like = not self.simple_like
    
    def get_cell_indexes(self,goal=False,transform=False):
        """gives the indexes of the cell where the goal or the robot is
        Boolean goal set to False to have robot information, True for the goal"""
        if goal:
            pos = self.goal_state.q
        else:
            pos = self.state.q
        ix,iy =  np.floor(pos[0]*self.grid_size), np.floor(pos[1]*self.grid_size)
        if transform:
            return self.grid_size*np.int(ix) + np.int(iy)
        return np.int(ix),np.int(iy)
    
    def available_path(self,depth_max=np.inf):
        """Computes all possible path of length depth_max starting from the robot position (BFS)
        One unit is equal to one cell length, specifies the paths whose length is equal to depth max"""
        ix,iy = self.get_cell_indexes()
        shift_table = {'N':-1,'S':1,'E':+self.grid_size,'W':-self.grid_size}
        directions = list(shift_table.keys())
        explored=[]
        path = [[self.grid_size*ix+iy]]
        to_treat= [[self.grid_size*ix+iy,0]]
        wanted_path = []
        while to_treat:
            current,path_idx = to_treat.pop(0)
            ix,iy = current//self.grid_size,current%self.grid_size
            for direction in directions:
                if self.maze.maze_map[ix][iy].walls[direction]==False:
                    children = current + shift_table[direction]
                    if not children in explored:
                        num = len(path)
                        current_path = path[path_idx].copy()
                        current_path.append(children)
                        if len(current_path) > depth_max:
                            continue
                        elif len(current_path) == depth_max:
                            wanted_path.append(num)
                            path.append(current_path)
                            to_treat.append([children,num])
                        else:
                            path.append(current_path)
                            to_treat.append([children,num])
            explored.append(current)
        return path,wanted_path
    
    def pick_goal_cell(self):
        """ randomly selects a cell whose distance to the robot follows the same distribution as in the simple maze."""
        values = np.array([2, 3, 4, 5, 6, 7, 8, 9]) 
        prob = np.array([0.0022 , 0.14768, 0.2606 , 0.2345 , 0.17393, 0.10553, 0.05643, 0.01913]) #histogram obtained by simulations
        depth = np.random.choice(values,p=prob)
        paths,wanted_path = self.available_path(depth_max=depth)
        N = len(wanted_path)
        i = np.random.randint(N)
        return paths[wanted_path[i]][-1]
    
    def shortest_path(self):
        """Computes the shortest path starting between the robot and the target in terms of cells"""
        ix,iy = self.get_cell_indexes()
        gx,gy = self.get_cell_indexes(goal=True)
        shift_table = {'N':-1,'S':1,'E':+self.grid_size,'W':-self.grid_size}
        directions = list(shift_table.keys())
        explored=[]
        goal = self.grid_size*gx+gy
        path = [[self.grid_size*ix+iy]]
        to_treat= [[self.grid_size*ix+iy,0]]
        while to_treat:
            current,path_idx = to_treat.pop(0)
            ix,iy = current//self.grid_size,current%self.grid_size
            for direction in directions:
                if self.maze.maze_map[ix][iy].walls[direction]==False:
                    children = current + shift_table[direction]
                    if not children in explored:
                        num = len(path)
                        current_path = path[path_idx].copy()
                        current_path.append(children)
                        path.append(current_path)
                        to_treat.append([children,num])
                        if children == goal:
                            return current_path
            explored.append(current)
        return False
    
    def get_direction(self,cell):
        """ computes the vector between the current position and the center of a target cell"""
        dind = (cell//self.grid_size +.5, cell%self.grid_size+.5)
        dpos = np.array([dind[0]/self.grid_size,dind[1]/self.grid_size])
        direction = dpos - self.state.q[:2]
        return direction
    
    def get_direction_to_goal(self):
        """ computes the vector between the current position and the goal"""
        return self.goal_state.q[:2]-self.state.q[:2]
    
    def set_within_margin(self,cell,action):
        """ In case of noisy actions, checks that the robot will be away from the walls."""
        idx,idy = cell//self.grid_size, cell%self.grid_size
        pos = self.state.q[:2]
        futur_pos = pos + action
        futur_pos[0] = max(min(futur_pos[0],idx/self.grid_size+1.5*self.thickness),(idx+1)/self.grid_size-1.5*self.thickness)
        futur_pos[1] = max(min(futur_pos[1],idy/self.grid_size+1.5*self.thickness),(idy+1)/self.grid_size-1.5*self.thickness)
        return futur_pos - pos
    
    def get_action(self,direction,cell,noisy):
        """ computes the most efficient action to reach a cell given the related direction, can add noise to the action"""
        if noisy:
            direction += ((np.random.random(2)-0.5)*0.30 +.1)*0.07
        action = direction*(np.abs(direction) <0.07) +\
        np.array([0.07,0.07])*(np.abs(direction) >0.07)*np.sign(direction)
        if noisy:
            action = self.set_within_margin(cell,action)
        closest= bool(np.prod(np.abs(action)<0.0105)) #to know if we can be closer
        action = action/0.07
        return action,closest
    
    def oracle(self,noisy=True):
        """ An oracle that performs the shortest safe path (i.e the one staying at the cells' center)"""
        solution = self.shortest_path()
        for desired_cell in solution:
            closest = False
            while not closest:
                direction = self.get_direction(desired_cell)
                action,closest = self.get_action(direction,desired_cell,noisy)
                states,reward,done,infos=self.step(action)
        closest = False
        while not closest:
            direction = self.get_direction_to_goal()
            action,closest = self.get_action(direction,None,False)
            states,reward,done,infos=self.step(action)
        return done[0]
    

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        self.maze = Maze(self.grid_size, self.grid_size)
        self.maze.make_maze()
        geom_objs = extract_obstacles(self.maze, self.thickness)
        geoms = Geometries(geom_objs)
        return geoms, idx_env

    def set_eval(self):
        pass

    def render(self, *unused_args, **unused_kwargs):
        if self.fig is None:
            self.init_matplotlib()
            self.pos = self.ax.scatter(
                self.state.q[0], self.state.q[1], color="orange", s=200
            )
        else:
            prev_pos = self.pos.get_offsets().data
            new_pos = self.state.q[:2][None, :]
            x = np.vstack((prev_pos, new_pos))
            self.ax.plot(x[:, 0], x[:, 1], c=(0, 0, 0, 0.5))
            self.pos.set_offsets(new_pos)
        plt.draw()
        plt.pause(0.01)

    def init_matplotlib(self):
        plt.ion()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_xlim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_ylim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_xticks([])
        ax.set_yticks([])

        obstacles = self.geoms.geom_objs
        rects = []
        for i, obst in enumerate(obstacles):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            rects.append(
                patches.Rectangle(
                    (x - w / 2, y - h / 2), w, h  # (x,y)  # width  # height
                )
            )
        coll = collections.PatchCollection(rects, zorder=1)
        coll.set_alpha(0.6)
        ax.add_collection(coll)

        size = self.robot_props["dist_goal"]
        offsets = np.stack((self.state.q, self.goal_state.q), 0)[:, :2]
        sg = collections.EllipseCollection(
            widths=size,
            heights=size,
            facecolors=[(1, 0, 0, 0.8), (0, 1, 0, 0.8)],
            angles=0,
            units="xy",
            offsets=offsets,
            transOffset=ax.transData,
        )
        ax.add_collection(sg)

        plt.tight_layout()
        self.fig = fig
        self.ax = ax


def extract_obstacles(maze, thickness):
    scx = 1 / maze.nx
    scy = 1 / maze.ny

    obstacles_coord = []
    for x in range(maze.nx):
        obstacles_coord.append((x / maze.nx, 0, (x + 1) / maze.nx, 0))
    for y in range(maze.ny):
        obstacles_coord.append((0, y / maze.ny, 0, (y + 1) / maze.ny))
    # Draw the "South" and "East" walls of each cell, if present (these
    # are the "North" and "West" walls of a neighbouring cell in
    # general, of course).
    for x in range(maze.nx):
        for y in range(maze.ny):
            if maze.cell_at(x, y).walls["S"]:
                x1, y1, x2, y2 = (
                    x * scx,
                    (y + 1) * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
            if maze.cell_at(x, y).walls["E"]:
                x1, y1, x2, y2 = (
                    (x + 1) * scx,
                    y * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
    obstacles = []
    for i, obst_coord in enumerate(obstacles_coord):
        x1, y1, x2, y2 = obst_coord[0], obst_coord[1], obst_coord[2], obst_coord[3]
        x1 -= thickness / 2
        x2 += thickness / 2
        y1 -= thickness / 2
        y2 += thickness / 2
        box_size = [x2 - x1, y2 - y1, 0.1]
        pos = [(x1 + x2) / 2, (y1 + y2) / 2, 0]
        placement = pin.SE3(np.eye(3), np.array(pos))
        mesh = Mesh(
            name=f"obstacle{i}",
            geometry=hppfcl.Box(*box_size),
            placement=placement,
            color=(0, 0, 1, 0.8),
        )
        obstacles.append(mesh.geom_obj())

    return obstacles


def maze_edges(grid_size):
    env = MazeGoal(grid_size)
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_raytracing(n_samples, n_rays):
    env = MazeGoal(grid_size=3)
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
