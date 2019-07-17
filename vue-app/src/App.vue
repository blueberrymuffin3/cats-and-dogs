<template lang="pug">
  v-app
    v-toolbar(app)
      v-toolbar-title(class="headline")
        | Cats and Dogs

    v-content
      v-container
        ModelUI
</template>

<script>
import ModelUI from "./components/ModelUI";

export default {
  name: "App",
  components: {
    ModelUI
  },
  data() {
    return {
      //
    };
  }
};
</script>
