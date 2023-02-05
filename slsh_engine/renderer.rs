use std::ffi::{c_char, CString};

use ash::vk;

use crate::window::Window;

pub struct Renderer {
    instance: ash::Instance,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = ash::Entry::linked();
        let req_exts = window.get_required_extensions();
        let req_exts_cstrs = convert_to_c_strs(&req_exts);
        let req_exts_cptrs = convert_to_c_ptrs(&req_exts_cstrs);
        let info = vk::InstanceCreateInfo::builder().enabled_extension_names(&req_exts_cptrs);

        let instance =
            unsafe { entry.create_instance(&info, None).expect("Failed to create instance") };

        Self { instance }
    }
}

fn convert_to_c_strs(strings: &[String]) -> Vec<CString> {
    strings
        .iter()
        .map(|string| CString::new(string.clone()).expect("String contains null byte"))
        .collect()
}

fn convert_to_c_ptrs(cstrings: &[CString]) -> Vec<*const c_char> {
    cstrings.iter().map(|cstring| cstring.as_ptr()).collect()
}
