pub mod ext_metabonk_control_v1 {
    pub mod server {
        use smithay::reexports::wayland_server;
        #[allow(unused_imports)]
        use smithay::reexports::wayland_server::protocol::*;

        pub mod __interfaces {
            #[allow(unused_imports)]
            use smithay::reexports::wayland_server::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-metabonk-control-v1.xml");
        }

        use self::__interfaces::*;
        wayland_scanner::generate_server_code!("protocols/ext-metabonk-control-v1.xml");
    }

    pub mod client {
        use wayland_client;
        #[allow(unused_imports)]
        use wayland_client::protocol::*;

        pub mod __interfaces {
            #[allow(unused_imports)]
            use wayland_client::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-metabonk-control-v1.xml");
        }

        use self::__interfaces::*;
        wayland_scanner::generate_client_code!("protocols/ext-metabonk-control-v1.xml");
    }
}

pub mod wl_drm {
    pub mod server {
        use smithay::reexports::wayland_server;
        #[allow(unused_imports)]
        use smithay::reexports::wayland_server::protocol::*;

        pub mod __interfaces {
            #[allow(unused_imports)]
            use smithay::reexports::wayland_server::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/wl-drm.xml");
        }

        use self::__interfaces::*;
        wayland_scanner::generate_server_code!("protocols/wl-drm.xml");
    }
}
