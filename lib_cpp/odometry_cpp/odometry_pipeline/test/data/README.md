# 3D odometry goldens (FARLIO-1010..1040)

`golden_3d_cpu.txt` was recorded on unmodified upstream `TUMFTM/FAR-LIO` `ef20510316ad1f8dbf2e428e4a193f700493eccd`
with `FARLIO_RECORD_GOLDEN=1 ./unit_3d_golden`, CPU build (`-DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`), in the
`deps` image `gr-far-lio-deps` = `sha256:76a9e31cdb61004f8519d4d95f19833a7c87fb0477ebbfadc4f25afa4e53a2fa`
(CUDA 12.8.1 devel, Ubuntu 24.04; the apt layer is unpinned). Two recordings were byte-identical.

The golden is compared bit-exact. It is a reviewed diff: regenerate it only in a commit that says why, never to make a
run pass. If an image move changes it, that is diagnosable from the digest above.
