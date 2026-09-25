// K200 rear blower shroud
// Card XY from measurement_canvas.js. +Z is off the 8-pin end, into the
// 170 x 100 mm case pocket (keep the Arctic front fans).
//
// Fan: Delta BFB1012VH. Drawing 97.2 x 94.4 x 33; measured snout 58 x 32.
// Outlet faces -Z into the cap, so 97.2 (blow + snail) is our Z and 94.4 is X.
// Duct is a straight extrusion of the heatsink window. No hull.
//
// PART: "flange" | "shroud" | "preview"

$fn = 64;
PART = "shroud";

eps        = 0.05;
wall       = 1.8;
flange_z   = 3.2;
hs_clear   = 0.7;
outer_grow = 0.8;
m2_d       = 2.3;
m2_head_d  = 4.8;
m2_head_z  = 1.6;
m2_boss_d  = 7.0;   // Ø4 mm pad; Ø5 was tight on the second flange print
m2_boss_z  = 0.6;   // into the flange from the card face
cable_z    = 32.0;  // match CASE.cable_run_z
cap_z      = 3.0;

// Straight HS duct through the cable run. Blower starts after the braid turns.
plenum_z   = cable_z;

// Intake face is 97.2 x 94.4. Blow is along 97.2 (= our Z); 94.4 fits in X.
blower_x   = 0.0;
blower_y   = 3.0;
blower_w   = 94.4;
blower_h   = 32.0;  // measured pack / snout; drawing is 33.0±0.5
blower_len = 97.2;

// Snout face, calipered 2026-09-20: 58 x 32 outer (drawing 57.9±0.5 x 33.0±0.5).
// Inner still from drawing wall stock (2.0 / 1.85); through-hole is ∩ HS window.
// Plan-view 50.0±0.5 is boxed snout depth along 97.2 (our Z), not the hole.
snout_w        = 58.0;
snout_h        = 32.0;
snout_inner_w  = 54.0;
snout_inner_h  = 28.3;
snout_stick    = 13.0;  // measured: grill face to snail body
snout_register = 2.0;   // plug depth into the cap (dead-ends below the HS)
snout_collar_z = snout_stick - snout_register - 3.0; // 8 mm tube; ~3 mm gap to snail
snout_x        = 8.0;   // over the wide HS band; flip 180° about Y for the other side
snout_y        = blower_y;
snout_clear    = 0.4;
snout_box_z    = 50.0;  // drawing: rectangular housing along the blow axis

outer_pts = [
    [0, 0], [97, 0], [97, 33], [81, 39], [16, 39], [0, 33]
];
hs_pts = [
    [16, 36.5], [81, 36.5], [81, 31], [91, 31], [91, 21],
    [51, 21], [51, 16], [6, 16], [6, 31], [16, 31]
];
inner_pts = [
    [16, 37], [81, 37], [93, 32], [93, 15], [88, 15], [88, 17],
    [65, 17], [65, 11], [4, 11], [4, 33]
];
screws = [[8, 11], [44, 11]];

module profile_outer() polygon(outer_pts);
module profile_inner() polygon(inner_pts);
module profile_hs() polygon(hs_pts);

module pin8_2d() {
    translate([62, -4]) square([50, 22]);
}

module hs_window() {
    intersection() {
        offset(delta=hs_clear) profile_hs();
        offset(delta=-0.25) profile_inner();
    }
}

module snout_outer_2d() {
    translate([snout_x - snout_clear, snout_y - snout_clear])
        square([snout_w + 2 * snout_clear, snout_h + 2 * snout_clear]);
}

module snout_inner_2d() {
    translate([
        snout_x + (snout_w - snout_inner_w) / 2,
        snout_y + (snout_h - snout_inner_h) / 2
    ]) square([snout_inner_w, snout_inner_h]);
}

module snout_air() {
    // Through the cap only where the snout overlaps the plenum — never the cable bay.
    intersection() {
        snout_inner_2d();
        hs_window();
    }
}

module blower_2d() {
    // Walls in Y only. 94.4 mm along X already fills the card.
    translate([blower_x, blower_y - wall])
        square([blower_w, blower_h + 2 * wall]);
}

module screw_through() {
    for (p = screws)
        translate([p[0], p[1], -2]) cylinder(d=m2_d, h=flange_z + 4);
}

module screw_counterbore() {
    for (p = screws)
        translate([p[0], p[1], flange_z - m2_head_z])
            cylinder(d=m2_head_d, h=m2_head_z + 0.4);
}

module screw_boss_clearance() {
    // Card-facing pockets for the Ø4 mm metal pads around the M2 holes.
    for (p = screws)
        translate([p[0], p[1], -1.2])
            cylinder(d=m2_boss_d, h=1.2 + m2_boss_z);
}

module locator_solid() {
    translate([0, 0, -1.0])
        difference() {
            linear_extrude(1.0) offset(delta=-0.45) profile_inner();
            linear_extrude(1.2) offset(delta=hs_clear + 0.4) profile_hs();
            linear_extrude(1.2) pin8_2d();
        }
}

module flange_plate() {
    linear_extrude(flange_z) offset(delta=outer_grow) profile_outer();
}

module pin8_cut() {
    translate([0, 0, -2]) linear_extrude(flange_z + 2) pin8_2d();
}

module hs_cavity() {
    // One hole through flange + duct. Stops short of the cap so the cap can close.
    translate([0, 0, -eps])
        linear_extrude(flange_z + plenum_z - 0.4 + eps)
            hs_window();
}

module duct_outer() {
    translate([0, 0, flange_z - eps])
        linear_extrude(plenum_z + cap_z + eps)
            offset(delta=wall) hs_window();
}

module end_cap() {
    z0 = flange_z + plenum_z;
    // Solid lid over the HS tube + a pad for the 32 mm snout (taller than the fins).
    translate([0, 0, z0 - eps])
        linear_extrude(cap_z + eps)
            union() {
                offset(delta=wall) hs_window();
                offset(delta=wall) snout_outer_2d();
            }
}

module snout_collar() {
    // Tube on the fan side of the cap. Snout slides in; snail stays ~3 mm off the rim.
    z0 = flange_z + plenum_z;
    translate([0, 0, z0 + cap_z - eps])
        linear_extrude(snout_collar_z + eps)
            offset(delta=wall) snout_outer_2d();
}

module fan_cuts() {
    z0 = flange_z + plenum_z;
    translate([0, 0, z0 - 1])
        linear_extrude(cap_z + snout_collar_z + 8)
            snout_air();
    // 58×32 plug through the collar and 2 mm into the cap; dead-ends below the HS.
    translate([0, 0, z0 + cap_z - snout_register])
        linear_extrude(snout_register + snout_collar_z + 0.4)
            snout_outer_2d();
}

module flange() {
    difference() {
        union() {
            flange_plate();
            locator_solid();
        }
        translate([0, 0, -eps]) linear_extrude(flange_z + 2 * eps) hs_window();
        pin8_cut();
        screw_through();
        screw_counterbore();
        screw_boss_clearance();
    }
}

module shroud() {
    difference() {
        union() {
            flange_plate();
            locator_solid();
            duct_outer();
            end_cap();
            snout_collar();
        }
        hs_cavity();
        pin8_cut();
        screw_through();
        screw_counterbore();
        screw_boss_clearance();
        fan_cuts();
    }
}

module snail_xz() {
    // 2D in XZ: y=0 is the snout face (toward the card), +y is +Z into the scroll.
    // Volute like the BFB1012 photo: round body, snout off to one side — not a box.
    cx = snout_x + snout_w / 2 + 8;
    cz = snout_box_z;
    difference() {
        intersection() {
            translate([blower_x, 0]) square([blower_w, blower_len]);
            union() {
                hull() {
                    translate([cx, cz]) circle(d=88);
                    translate([snout_x, 3]) square([snout_w, 8]);
                }
                translate([snout_x, -snout_stick])
                    square([snout_w, snout_stick + 10]);
            }
        }
        translate([cx, cz]) circle(d=74);
        translate([
            snout_x + (snout_w - snout_inner_w) / 2,
            -snout_stick - 1
        ]) square([snout_inner_w, snout_stick + 18]);
    }
}

module preview_blower() {
    // BFB1012VH snail. 32 mm in Y (slot). Intake is the round face (±Y).
    // Snout points -Z into the cap. Look from +Y to see the snail; from +Z
    // you only see the 32 mm edge and the 58×32 mouth (orange frame).
    z_cap    = flange_z + plenum_z;
    // Grill sits in the 2 mm cap register; snail is snout_stick behind that face.
    snout_z0 = z_cap + cap_z - snout_register + snout_stick;

    color([0.18, 0.18, 0.22, 0.72])
        translate([0, blower_y + blower_h, snout_z0])
            rotate([90, 0, 0])
                linear_extrude(blower_h, convexity=10)
                    snail_xz();

    // Mouth frame — 58×32 register; through-hole is only the overlap with the HS.
    color([0.85, 0.35, 0.1, 0.9])
        translate([snout_x, snout_y, z_cap])
            difference() {
                cube([snout_w, snout_h, 1.2]);
                translate([0, 0, -0.2])
                    linear_extrude(2) snout_air();
            }
}

module preview_envelope() {
    color([0.2, 0.6, 0.3, 0.12])
        translate([-1.5, 0, 0]) cube([100, 39, 170]);
}

module preview_card() {
    color([0.45, 0.45, 0.5, 0.3]) linear_extrude(0.4) profile_outer();
    color([0.85, 0.15, 0.1, 0.4])
        translate([0, 0, 0.4]) linear_extrude(0.4) profile_hs();
    color([0.1, 0.1, 0.1])
        for (p = screws) translate([p[0], p[1], -0.2]) cylinder(d=2, h=1);
}

if (PART == "flange") {
    flange();
} else if (PART == "preview") {
    preview_envelope();
    preview_card();
    color([0.15, 0.5, 0.85, 0.75]) shroud();
    preview_blower();
} else {
    shroud();
}
