"""Waterhose-specific VBD solver hooks."""

from __future__ import annotations

import newton
import warp as wp
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.rigid_vbd_kernels import accumulate_body_particle_contact_forces_on_proxy_bodies


@wp.kernel(enable_backward=False)
def _harvest_proxy_normal_wrenches_kernel(
    rigid_contact_count: wp.array[int],
    contact_body0: wp.array[wp.int32],
    contact_body1: wp.array[wp.int32],
    contact_point0_world: wp.array[wp.vec3],
    contact_point1_world: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_force_on_body1: wp.array[wp.vec3],
    dst_body_inv_mass: wp.array[float],
    dst_body_flags: wp.array[wp.int32],
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    out_proxy_body_f: wp.array[wp.spatial_vector],
):
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    body0 = contact_body0[contact_id]
    body1 = contact_body1[contact_id]
    if body0 < 0 or body1 < 0:
        return

    is_proxy0 = int(0)
    is_proxy1 = int(0)
    proxy_global0 = int(-1)
    proxy_global1 = int(-1)
    if body0 < dst_body_flags.shape[0] and (dst_body_flags[body0] & proxy_flag) != 0:
        proxy_global0 = body_local_to_proxy_global[body0]
        if proxy_global0 >= 0:
            is_proxy0 = 1
    if body1 < dst_body_flags.shape[0] and (dst_body_flags[body1] & proxy_flag) != 0:
        proxy_global1 = body_local_to_proxy_global[body1]
        if proxy_global1 >= 0:
            is_proxy1 = 1

    if (is_proxy0 + is_proxy1) != 1:
        return

    other_id = body1 if is_proxy0 == 1 else body0
    if other_id < 0 or other_id >= dst_body_inv_mass.shape[0]:
        return
    if dst_body_inv_mass[other_id] <= 0.0:
        return

    force_on_body1 = contact_force_on_body1[contact_id]
    if is_proxy1 == 1:
        proxy_local_id = body1
        proxy_global_id = proxy_global1
        contact_point = contact_point1_world[contact_id]
        force_on_proxy = force_on_body1
    else:
        proxy_local_id = body0
        proxy_global_id = proxy_global0
        contact_point = contact_point0_world[contact_id]
        force_on_proxy = -force_on_body1

    if proxy_global_id < 0 or proxy_global_id >= out_proxy_body_f.shape[0]:
        return

    normal = contact_normal[contact_id]
    normal_length = wp.length(normal)
    if normal_length > 1.0e-8:
        normal = normal / normal_length
        force_on_proxy = wp.dot(force_on_proxy, normal) * normal

    com_world = wp.transform_point(body_q[proxy_local_id], body_com[proxy_local_id])
    torque = wp.cross(contact_point - com_world, force_on_proxy)
    wp.atomic_add(out_proxy_body_f, proxy_global_id, wp.spatial_vector(force_on_proxy, torque))


class WaterhoseSolverVBD(SolverVBD):
    """VBD solver that removes tangential proxy feedback."""

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array,
        out_body_f: wp.array,
        *,
        state=None,
        state_out=None,
        contacts=None,
        dt: float = 0.0,
    ) -> None:
        """Harvest contact-only proxy-body wrenches without tangential feedback."""
        del state
        if not self._coupling_has_rigid_avbd_state:
            raise NotImplementedError("VBD proxy contact harvest requires rigid-body AVBD state")

        if contacts is None or state_out is None:
            return

        body_q_prev = self._coupling_proxy_body_q_prev

        if contacts.rigid_contact_max > 0:
            body0, body1, point0, point1, force_on_body1, rigid_contact_count = self.collect_rigid_contact_forces(
                state_out.body_q,
                body_q_prev,
                contacts,
                dt,
            )
            wp.launch(
                _harvest_proxy_normal_wrenches_kernel,
                dim=contacts.rigid_contact_max,
                inputs=[
                    rigid_contact_count,
                    body0,
                    body1,
                    point0,
                    point1,
                    contacts.rigid_contact_normal,
                    force_on_body1,
                    self.model.body_inv_mass,
                    self.model.body_flags,
                    body_local_to_proxy_global,
                    int(newton.BodyFlags.PROXY),
                    self.model.body_com,
                    state_out.body_q,
                    out_body_f,
                ],
                device=self.device,
            )

        if contacts.soft_contact_max > 0 and self.body_particle_contact_penalty_k.shape[0] >= contacts.soft_contact_max:
            wp.launch(
                accumulate_body_particle_contact_forces_on_proxy_bodies,
                dim=contacts.soft_contact_max,
                inputs=[
                    float(dt),
                    body_local_to_proxy_global,
                    state_out.particle_q,
                    self.particle_q_prev,
                    self.model.particle_radius,
                    state_out.body_q,
                    body_q_prev,
                    self.model.body_com,
                    float(self.friction_epsilon),
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.shape_body,
                    out_body_f,
                ],
                device=self.device,
            )
