"""
CPE (Cylindrical Parabolic Equation) — 2D FFT + DMFT + TWPE.

Forward PE:  thesis Eq. (2-36)
Backward PE: thesis Eq. (5-2)
Multi-bounce iteration with convergence threshold ε (thesis Eq. 5-4).
"""

import numpy as np
from scipy.special import hankel1 as _hankel1
from numba import njit

from src.antenna_types import make_initial_field, DEFAULT_ANTENNA_CONFIG

C0 = 2.99792458e8; EPS0 = 8.854187817e-12; N_ATM = 1.0003
SIGMA_CONDUCTOR = 1e5
# ── TWPE (双向抛物方程) — 暂不使用，性能开销大 ──
# TWPE_EPSILON = 1e-3
# TWPE_MAX_ITER = 5

N_Z = 2048; N_PHI = 128; DR_FACTOR = 1.0
Z_PAD_ABOVE = 20.0; R0_FACTOR = 2.0


def material_n(freq, eps_r, sigma):
    omega = 2.0 * np.pi * freq
    return np.sqrt(eps_r - 1j * sigma / (omega * EPS0 + 1e-30))

def is_conductor(sigma):
    return sigma > SIGMA_CONDUCTOR


class CPESolver2D:
    def __init__(self, frequency, antenna_pos, scene_objects,
                 n_atm=N_ATM, n_z=N_Z, n_phi=N_PHI, dr_factor=DR_FACTOR):
        self.freq = float(frequency); self.antenna = np.asarray(antenna_pos, dtype=np.float32)
        self.scene = scene_objects; self.n_atm = n_atm; self.n_z = n_z; self.n_phi = n_phi
        self.dr_factor = dr_factor
        self.k0 = 2.0 * np.pi * self.freq / C0; self.wavelength = C0 / self.freq
        self.phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.m_vals = np.fft.fftfreq(n_phi, 1.0 / n_phi).astype(int)

    def compute(self, rx_pos):
        rx = np.asarray(rx_pos, dtype=np.float32)
        rt, pt = self._cart_to_cyl(rx)
        r_vals, dr, nr = self._build_range_grid(rt)
        z_vals, dz = self._build_height_grid(rx[2])
        self._last_z_max = z_vals[-1]
        z_grd, z_top, n_grd, c_grd, n_lay, c_lay = self._extract_material_maps(nr, r_vals)
        dmft = self._build_dmft(c_grd)

        u_init = self._init_field(z_vals)

        u_fwd = self._pe_march(u_init.copy(), r_vals, dr, z_vals, dz,
                                z_grd, z_top, n_grd, c_grd, n_lay, c_lay,
                                dmft, forward=True)

        # ── TWPE: backward reflections (暂不使用) ──
        # for iteration in range(TWPE_MAX_ITER):
        #     sources = self._find_reflection_sources(u_total, r_vals, z_vals, dz,
        #                                              z_top, c_lay)
        #     if not sources:
        #         break
        #     u_accum = np.zeros_like(u_total)
        #     for src in sources:
        #         ri_start = src["r_idx"]
        #         r_sub = r_vals[:ri_start + 1]
        #         u_start = src["field"]
        #         u_bwd = self._pe_march(u_start, r_sub[::-1], -dr, z_vals, dz,
        #                                 z_grd[:, :ri_start+1][:, ::-1],
        #                                 z_top[:, :ri_start+1][:, ::-1],
        #                                 n_grd[:, :ri_start+1][:, ::-1],
        #                                 c_grd[:, :ri_start+1][:, ::-1],
        #                                 n_lay[:, :ri_start+1][:, ::-1],
        #                                 c_lay[:, :ri_start+1][:, ::-1],
        #                                 dmft, forward=False)
        #         u_bwd_full = np.zeros_like(u_total)
        #         u_bwd_full[:ri_start+1] = u_bwd[::-1]
        #         u_accum += u_bwd_full
        #     u_total = u_fwd + u_accum
        #     change = np.linalg.norm(u_total - prev_total) / (np.linalg.norm(prev_total) + 1e-15)
        #     if change < TWPE_EPSILON:
        #         break
        #     prev_total = u_total.copy()

        u_total = u_fwd  # TWPE 暂不使用，直接取前向结果

        E_rx, E_fs = self._extract(u_total, rt, pt, rx[2], r_vals, z_vals)
        L_fs = 20 * np.log10(self.freq/1e6) + 20 * np.log10(rt/1000) + 32.45
        L_pe = L_fs - 20 * np.log10(max(E_rx / E_fs, 1e-12))
        return {"path_loss_dB": float(L_pe), "E_rx": float(E_rx), "E_fs": float(E_fs),
                "L_fs_dB": float(L_fs), "r_grid": r_vals, "z_grid": z_vals,
                "phi_grid": self.phi_vals, "z_terrain_2d": z_top}

    def _cart_to_cyl(self, rx):
        dx, dy = rx[0]-self.antenna[0], rx[1]-self.antenna[1]
        r = np.hypot(dx, dy); phi = np.arctan2(dy, dx)
        if phi < 0: phi += 2*np.pi
        return r, phi

    def _build_range_grid(self, rt):
        r0 = R0_FACTOR * self.wavelength; dr = self.dr_factor * self.wavelength
        nr = max(int((rt-r0)/dr)+1, 2); return np.linspace(r0, rt, nr), dr, nr

    def _build_height_grid(self, z_rx):
        z_max = max(z_rx, self.antenna[2]) + Z_PAD_ABOVE
        dz = z_max / (self.n_z - 1); return np.linspace(0, z_max, self.n_z), dz

    def _init_field(self, z_vals):
        ant_obj = self.scene.get("antenna", {})
        ant_config = ant_obj.get("extra", {}).get("antenna_config", dict(DEFAULT_ANTENNA_CONFIG))
        h_ant = self.antenna[2]
        r0 = R0_FACTOR * self.wavelength
        return make_initial_field(ant_config, z_vals, self.n_phi, h_ant, self.k0, r0)

    # ── DMFT ──
    def _build_dmft(self, cond_map):
        mat = _get_ground_mat(self.scene)
        if mat is None or is_conductor(mat["sigma"]): return None
        n_c = material_n(self.freq, mat["eps_r"], mat["sigma"])
        alpha = 1j * self.k0 * np.sqrt(n_c**2 - 1.0) / (n_c**2)
        return self._make_dmft(alpha)

    def _make_dmft(self, alpha):
        nz = self.n_z; z_max = self._last_z_max; dz = z_max/(nz-1)
        z_vals = np.linspace(0, z_max, nz)
        k_n = (np.arange(nz)+0.5)*np.pi/(z_max+dz)
        kzg, zg = k_n[:,None], z_vals[None,:]
        phi_nz = np.cos(kzg*zg)+(alpha/(kzg+1e-30))*np.sin(kzg*zg)
        N_n = (z_max/2.0)*(1.0+np.abs(alpha/(k_n+1e-30))**2)
        F = phi_nz*(dz/N_n[:,None]); G = phi_nz.T
        self._dmft_kz = k_n; return {"F":F,"G":G,"kz":k_n}

    # ── PE march (forward or backward) ──
    def _pe_march(self, u, r_vals, dr, z_vals, dz,
                  z_grd, z_top, n_grd, c_grd, n_lay, c_lay, dmft, forward):
        nr = len(r_vals); nz = self.n_z; nphi = self.n_phi
        if dmft is not None: kz_vals = dmft["kz"]
        else: kz_vals = 2.0*np.pi*np.fft.fftfreq(nz, dz)
        k_r = np.sqrt(np.maximum((self.k0*self.n_atm)**2-kz_vals**2, 0.0))
        k_r_ok = k_r > 1e-12
        refract = np.exp(1j*self.k0*(self.n_atm-2.0)*abs(dr))
        n_taper = nz//8; taper = np.ones(nz)
        for k in range(n_taper):
            taper[nz-n_taper+k] = 0.5+0.5*np.cos(8.0*np.pi*k/nz)
        g_idx = np.clip(((z_grd)/dz).astype(int), 0, nz)  # ground surface
        t_idx = np.clip(((z_top)/dz).astype(int), 0, nz)  # top of clip layer
        u_full = np.zeros((nr, nphi, nz), dtype=np.complex64); u_full[0] = u

        for i in range(1, nr):
            rp, rc = r_vals[i-1], r_vals[i]
            if dmft is not None: Uz = dmft["F"] @ u.T
            else: Uz = np.fft.fft(u, axis=1).T
            for ki in range(nz):
                if not k_r_ok[ki]: Uz[ki,:]=0.0; continue
                Um = np.fft.fft(Uz[ki,:])
                xp, xc = k_r[ki]*rp, k_r[ki]*rc
                hp = _hankel1(self.m_vals, xp); hc = _hankel1(self.m_vals, xc)
                Um *= np.where(np.abs(hp)>1e-15, hc/hp, 0.0)
                Uz[ki,:] = np.fft.ifft(Um)
            if dmft is not None: u = (dmft["G"] @ Uz).T
            else: u = np.fft.ifft(Uz.T, axis=1)
            u *= refract
            if min(rp,rc)>0: u *= np.sqrt(rp/rc)
            # Two-layer terrain boundary
            for pi in range(nphi):
                gi = g_idx[pi, i]
                ti = t_idx[pi, i]
                if gi <= 0: continue
                # Ground layer (below terrain surface)
                if c_grd[pi, i]:
                    u[pi, :gi] = 0.0
                else:
                    u[pi, :gi] *= np.exp(1j*self.k0*(n_grd[pi,i]-self.n_atm)*abs(dr))
                # Clip layer (between ground and clip top)
                if ti > gi:
                    if c_lay[pi, i]:
                        u[pi, gi:ti] = 0.0
                    else:
                        u[pi, gi:ti] *= np.exp(1j*self.k0*(n_lay[pi,i]-self.n_atm)*abs(dr))
            u[:,:] *= taper[np.newaxis,:]; u_full[i] = u
        return u_full

    # ── Reflection source detection ──
    def _find_reflection_sources(self, u_total, r_vals, z_vals, dz, z_top, c_lay):
        nr = len(r_vals)
        # Accumulate all φ contributions at each r_idx
        src_by_r = {}
        for pi in range(self.n_phi):
            zt = z_top[pi, :]
            diffs = np.diff(zt)
            jumps = np.where(np.abs(diffs) > dz)[0] + 1
            for j in jumps:
                if not c_lay[pi, j]: continue
                ri = min(j, nr-1)
                ti = int(zt[j] / dz) if zt[j] < len(z_vals)*dz else 0
                if ti <= 0: continue
                if ri not in src_by_r:
                    src_by_r[ri] = np.zeros((self.n_phi, self.n_z), dtype=np.complex64)
                src_by_r[ri][pi, :ti] = -u_total[ri, pi, :ti]
        return [{"r_idx": ri, "field": ub} for ri, ub in src_by_r.items()]

    def _extract(self, u_full, rt, pt, z_rx, r_vals, z_vals):
        ri = len(r_vals)-1; pi = np.argmin(np.abs(self.phi_vals-pt))
        zi = np.argmin(np.abs(z_vals-z_rx))
        E_rx = np.abs(u_full[ri, pi, zi])
        E_fs = r_vals[0]/max(rt,1e-6); return E_rx, E_fs

    # ── Terrain + materials ──
    def _extract_material_maps(self, nr, r_vals):
        # z_ground = base terrain surface; z_top = top of any clip layer
        zg=np.zeros((self.n_phi,nr),dtype=np.float32); zt=np.zeros((self.n_phi,nr),dtype=np.float32)
        # ng/cm = ground material below surface
        ng=np.full((self.n_phi,nr),self.n_atm,dtype=np.complex64)
        cg=np.zeros((self.n_phi,nr),dtype=bool)
        # nl/cl = clip layer material (on top of ground), or same as ground if no clip
        nl=np.full((self.n_phi,nr),self.n_atm,dtype=np.complex64)
        cl=np.zeros((self.n_phi,nr),dtype=bool)
        for pi,phi in enumerate(self.phi_vals):
            c,s=np.cos(phi),np.sin(phi); xp=self.antenna[0]+r_vals*c; yp=self.antenna[1]+r_vals*s
            t=self.scene.get("terrain")
            if t is not None:
                gz=_sample_terrain(t["mesh"],xp,yp)
                zg[pi,:]=gz; zt[pi,:]=gz
                m=_get_mat(t)
                if m is not None:
                    nc,_=_resolve_mat(self.freq,m); ng[pi,:]=nc; nl[pi,:]=nc
                    cg[pi,:]=is_conductor(m["sigma"]); cl[pi,:]=cg[pi,:]
            # ── Wall obstacles: per‑phi blocking with minimum shadow width ──
            for nm_key, o in self.scene.items():
                extra = o.get("extra") or {}
                if extra.get("obstacle_type") != "wall":
                    continue
                msh = o.get("mesh")
                if msh is None: continue
                wb = msh.bounds
                wm = _get_mat(o)
                if wm is not None:
                    wnc, wcond = _resolve_mat(self.freq, wm)
                else:
                    wnc, wcond = np.complex64(self.n_atm), True
                for pi2, phi2 in enumerate(self.phi_vals):
                    c2, s2 = np.cos(phi2), np.sin(phi2)
                    xp2 = self.antenna[0] + r_vals * c2
                    yp2 = self.antenna[1] + r_vals * s2
                    inside2 = ((xp2 >= wb[0]) & (xp2 <= wb[1]) &
                               (yp2 >= wb[2]) & (yp2 <= wb[3]))
                    if not inside2.any():
                        continue
                    # 扩展至最小 3 步的径向段
                    idx_list = np.where(inside2)[0]
                    if len(idx_list) < 3:
                        mid = idx_list[len(idx_list) // 2]
                        idx_list = np.arange(max(0, mid - 1), min(nr, mid + 2))
                    zg[pi2, idx_list] = np.maximum(zg[pi2, idx_list], wb[5])
                    zt[pi2, idx_list] = np.maximum(zt[pi2, idx_list], wb[5])
                    ng[pi2, idx_list] = wnc; nl[pi2, idx_list] = wnc
                    cg[pi2, idx_list] = wcond; cl[pi2, idx_list] = wcond

            # Standard obstacles — skip wall‑type (already handled)
            for nm_key, o in self.scene.items():
                if nm_key in ("terrain", "antenna") or "_clip" in nm_key:
                    continue
                if nm_key.startswith("layer_"):
                    continue
                if (o.get("extra") or {}).get("obstacle_type") == "wall":
                    continue
                if o.get("type")!="mesh": continue
                msh=o["mesh"]
                if msh is None: continue
                b=msh.bounds
                inside=((xp>=b[0])&(xp<=b[1])&(yp>=b[2])&(yp<=b[3]))
                if inside.any():
                    zg[pi,inside]=np.maximum(zg[pi,inside],b[5])
                    zt[pi,inside]=np.maximum(zt[pi,inside],b[5])
                    m=_get_mat(o)
                    if m is not None:
                        nc,_=_resolve_mat(self.freq,m); ng[pi,inside]=nc; cg[pi,inside]=is_conductor(m["sigma"])
                        nl[pi,inside]=nc; cl[pi,inside]=cg[pi,inside]
            # Clip layers — add thickness on top, keep ground below
            clips=[(k,o) for k,o in self.scene.items() if "_clip" in k]
            clips.sort(key=lambda x:_clip_counter(x[0]))
            for nm_key,o in clips:
                msh=o["mesh"]
                if msh is None: continue
                extra=o.get("extra",{}); polys=extra.get("polygons",[]); m=extra.get("material")
                thick=m.get("thickness_cm",0.0)/100.0 if m else 0.0
                if polys:
                    for ri in range(len(xp)):
                        if any(_point_in_poly(xp[ri],yp[ri],p) for p in polys):
                            zt[pi,ri]=zg[pi,ri]+thick
                            if m is not None:
                                nc,_=_resolve_mat(self.freq,m); nl[pi,ri]=nc; cl[pi,ri]=is_conductor(m["sigma"])
                else:
                    b=msh.bounds
                    inside=((xp>=b[0])&(xp<=b[1])&(yp>=b[2])&(yp<=b[3]))
                    if inside.any():
                        zt[pi,inside]=zg[pi,inside]+thick
                        if m is not None:
                            nc,_=_resolve_mat(self.freq,m); nl[pi,inside]=nc; cl[pi,inside]=is_conductor(m["sigma"])
        return zg,zt,ng,cg,nl,cl


def _clip_counter(name):
    """Extract counter from clip layer name (e.g., 'layer_water_clip_3' → 3)."""
    parts = name.split("_clip_")
    if len(parts) > 1:
        try: return int(parts[1])
        except ValueError: pass
    return 0


@njit
def _point_in_poly(x, y, poly):
    """Ray-casting point-in-polygon test."""
    n = len(poly); inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i,1] > y) != (poly[j,1] > y)) and \
           (x < (poly[j,0] - poly[i,0]) * (y - poly[i,1]) / (poly[j,1] - poly[i,1] + 1e-30) + poly[i,0]):
            inside = not inside
        j = i
    return inside


def _get_mat(obj):
    e=obj.get("extra"); return e.get("material") if e is not None else None

def _get_ground_mat(scene):
    t=scene.get("terrain"); return _get_mat(t) if t is not None else None

def _resolve_mat(freq,mat):
    return material_n(freq,mat["eps_r"],mat["sigma"]),is_conductor(mat["sigma"])

def _sample_terrain(mesh, xp, yp):
    pts = np.asarray(mesh.points, dtype=np.float32)
    if pts.shape[1] < 3:
        return np.zeros_like(xp, dtype=np.float32)
    try:
        dims = mesh.dimensions
        nx, ny = dims[0], dims[1]
        xs = pts[:nx, 0]
        ys = pts[::nx, 1][:ny]
        z2d = np.ascontiguousarray(pts[:, 2].reshape((ny, nx), order="F").astype(np.float32))
        return _bilinear_interp_terrain(xs, ys, z2d, np.asarray(xp, dtype=np.float32), np.asarray(yp, dtype=np.float32))
    except Exception:
        return _nearest_terrain(pts, np.asarray(xp, dtype=np.float32), np.asarray(yp, dtype=np.float32))


@njit
def _bilinear_interp_terrain(xs, ys, z2d, xp, yp):
    nx = z2d.shape[1]; ny = z2d.shape[0]
    r = np.empty(len(xp), dtype=np.float32)
    for i in range(len(xp)):
        x, y = xp[i], yp[i]
        ix = max(0, min(np.searchsorted(xs, x) - 1, nx - 2))
        iy = max(0, min(np.searchsorted(ys, y) - 1, ny - 2))
        x1, x2 = xs[ix], xs[ix + 1]
        y1, y2 = ys[iy], ys[iy + 1]
        dx = (x - x1) / (x2 - x1) if x2 != x1 else 0.5
        dy = (y - y1) / (y2 - y1) if y2 != y1 else 0.5
        r[i] = (z2d[iy, ix] * (1.0 - dx) * (1.0 - dy) +
                z2d[iy, ix + 1] * dx * (1.0 - dy) +
                z2d[iy + 1, ix] * (1.0 - dx) * dy +
                z2d[iy + 1, ix + 1] * dx * dy)
    return r


@njit
def _nearest_terrain(pts, xp, yp):
    r = np.empty(len(xp), dtype=np.float32)
    for i in range(len(xp)):
        x, y = xp[i], yp[i]
        best = 1e30; best_z = 0.0
        for j in range(len(pts)):
            d = (pts[j, 0] - x) * (pts[j, 0] - x) + (pts[j, 1] - y) * (pts[j, 1] - y)
            if d < best:
                best = d; best_z = pts[j, 2]
        r[i] = best_z
    return r
