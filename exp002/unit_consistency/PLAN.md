# Source-unit consistency development variant

Written after the first development failure; no confirmation seed is used.

The 2011 listed fixed physical weights have mean .1 on their nonzero supports:
singletons .1 and three-item lists .15/.1/.05. They are divided by list size in
the field's bottom-up current. The 2014 instar memory instead sums to one per
category. Passing it directly through the 2011 size divisor introduces a second
size normalization. This was an incompatibility in our declared synthesis,
not a defect demonstrated in either source model.

Test the explicit conversion physical_W=.1*size*normalized_W, likewise M.
This recovers the listed .1 singleton and mean .1 larger-list scale without
assigning any correct sequence prototype. Store and learn normalized W/M;
convert only the currents. This is a new model variant, not a recovered author
implementation or complete resolution of source dynamics.

The bounded check uses the same three DEVELOPMENT seeds and the same six
pulse/feedback settings as the first source probe, with the same 160 training
trials per setting. No parameter is selected by stability or robustness gain.
If selective learned list competition is still absent, retain the stop.
A passing variant would still require fair recurrent-control optimization,
headroom calibration, and a new executable frozen protocol before confirmation.
